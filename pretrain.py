"""Step 3 of CRIST: Stage-1 composition-guided multimodal pretraining on the pseudo-spots.

Model (BranchedProportionDeconvModel, as in the original proportion_deconv_branched.py):
  gene branch       log1p transcript counts (D_g)  -> MLP 512, 256 -> 128
  morphology branch nucleus tokens (n_i x 1280)    -> linear 256 -> attention pooling (4 heads, one learnable query) -> 256
  fusion head       [gene 128 | morphology 256 | n_i / 100] = 385 -> MLP 256, 128 -> softmax over the K cell types
  auxiliary heads   gene_head 128 -> 64 -> K and morph_head 256 -> 64 -> K (softmax), weighted by w_aux
Hidden layers of the gene branch and the fusion head use BatchNorm, ReLU and dropout 0.3; the auxiliary heads use
ReLU and dropout only. Nucleus tokens are L2-normalised; a spot without nuclei feeds one zero token with n_i = 0.
Loss on every head: w_pcc * (1 - PCC) + w_rmse * RMSE + w_jsd * JSD, each computed per cell type across the minibatch.
Model selection: the epoch with the best validation score PCC - RMSE - JSD; early stopping on that score.
"""
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SPOT_RADIUS_UM = 27.5      # a nucleus belongs to the spot whose centre is within this radius (nearest one if several)
MAX_CELLS_PER_SPOT = 200   # nuclei per spot are randomly subsampled beyond this (never reached in the paper's samples)


# =============================================================================
# Spot-level inputs
# =============================================================================

def spot_table(adata, cell_types):
    """log1p expression, composition labels and raw cell counts of a step-1 pseudo-spot AnnData."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    counts = adata.obs[[f"n_{ct}" for ct in cell_types]].to_numpy(dtype=np.float32)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)          # spots without cells keep an all-zero label
    return {
        "gene_expression": np.log1p(X.astype(np.float64)).astype(np.float32),
        "gene_names": list(adata.var_names),
        "labels": counts / row_sums,                                # proportions (sum = 1)
        "raw_counts": counts,
        "spot_ids": list(adata.obs_names),
        "spot_xy": np.asarray(adata.obsm["spatial"], dtype=np.float64),
    }


def align_genes(expression, gene_names, reference_genes):
    """Reorder columns to `reference_genes`; genes the sample does not measure become zero columns."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    aligned = np.zeros((expression.shape[0], len(reference_genes)), dtype=np.float32)
    for i, gene in enumerate(reference_genes):
        if gene in gene_to_idx:
            aligned[:, i] = expression[:, gene_to_idx[gene]]
    return aligned


def map_nuclei_to_spots(positions_px, spot_xy_um, mpp, radius=SPOT_RADIUS_UM):
    """Assign every CellViT nucleus (H&E pixels) to the spot whose centre lies within `radius` um of it.

    A nucleus within the radius of several centres goes to the nearest one; a nucleus farther than
    `radius` from every centre (the corners of the hexagonal cells) is not assigned to any spot.
    Returns a list with the nucleus indices of every spot.
    """
    positions_um = np.asarray(positions_px, dtype=np.float64) * mpp
    tree = cKDTree(spot_xy_um)
    spot_to_cells = [[] for _ in range(len(spot_xy_um))]
    for cell_idx, pos in enumerate(positions_um):
        spots = tree.query_ball_point(pos, r=radius)
        if len(spots) == 1:
            spot_to_cells[spots[0]].append(cell_idx)
        elif len(spots) > 1:
            distances = np.linalg.norm(spot_xy_um[spots] - pos, axis=1)
            spot_to_cells[spots[int(np.argmin(distances))]].append(cell_idx)
    return spot_to_cells


def build_source(pseudo, nuclei, cell_types, mpp):
    """Combine the pretraining samples into one training set.

    pseudo: {sample: pseudo-spot AnnData (step 1)}, nuclei: {sample: dict from nuclei_segmentation.load_cells (step 2)}.
    The model gene space is the intersection of the samples' feature names (sorted).
    """
    tables = {name: spot_table(adata, cell_types) for name, adata in pseudo.items()}
    common_genes = sorted(set.intersection(*[set(t["gene_names"]) for t in tables.values()]))
    embeddings, spot_to_cells, offset = [], [], 0
    for name, table in tables.items():
        cells = nuclei[name]
        mapping = map_nuclei_to_spots(cells["positions"].numpy(), table["spot_xy"], mpp)
        spot_to_cells += [[i + offset for i in cell_list] for cell_list in mapping]
        embeddings.append(cells["embeddings"])
        offset += cells["embeddings"].shape[0]
        print(f"{name}: {len(table['spot_ids']):,} spots, {sum(len(c) for c in mapping):,} of "
              f"{len(cells['positions']):,} nuclei assigned to a spot")
    return {
        "gene_expression": np.concatenate([align_genes(t["gene_expression"], t["gene_names"], common_genes) for t in tables.values()]),
        "labels": np.concatenate([t["labels"] for t in tables.values()]),
        "raw_counts": np.concatenate([t["raw_counts"] for t in tables.values()]),
        "cell_embeddings": torch.cat(embeddings, dim=0),
        "spot_to_cells": spot_to_cells,
        "common_genes": common_genes,
        "cell_types": list(cell_types),
    }


def build_target(adata, cells, cell_types, reference_genes, mpp):
    """One sample aligned to the model gene space (for monitoring in step 3 and adaptation in step 5)."""
    table = spot_table(adata, cell_types)
    mapping = map_nuclei_to_spots(cells["positions"].numpy(), table["spot_xy"], mpp)
    print(f"{adata.uns['pseudo_spots']['sample']}: {len(table['spot_ids']):,} spots, "
          f"{sum(len(c) for c in mapping):,} of {len(cells['positions']):,} nuclei assigned to a spot; "
          f"{len(set(table['gene_names']) & set(reference_genes))} of {len(reference_genes)} model genes measured")
    return {
        "gene_expression": align_genes(table["gene_expression"], table["gene_names"], reference_genes),
        "labels": table["labels"],
        "raw_counts": table["raw_counts"],
        "cell_embeddings": cells["embeddings"],
        "spot_to_cells": mapping,
        "spot_ids": table["spot_ids"],
        "cell_types": list(cell_types),
    }


class SpatialMultimodalDataset(Dataset):
    """One item = the log1p expression of a spot, the (L2-normalised) tokens of its nuclei and its label."""

    def __init__(self, gene_expression, cell_embeddings, spot_to_cells, labels, max_cells=MAX_CELLS_PER_SPOT):
        self.gene_expression = torch.FloatTensor(gene_expression)
        self.cell_embeddings = F.normalize(cell_embeddings, p=2, dim=1)
        self.spot_to_cells = spot_to_cells
        self.labels = torch.FloatTensor(labels)
        self.max_cells = max_cells

    def __len__(self):
        return len(self.gene_expression)

    def __getitem__(self, idx):
        cell_indices = self.spot_to_cells[idx]
        if len(cell_indices) == 0:
            cell_embs, n_cells = torch.zeros((1, self.cell_embeddings.shape[1])), 0   # placeholder token
        else:
            if len(cell_indices) > self.max_cells:
                cell_indices = random.sample(cell_indices, self.max_cells)
            cell_embs, n_cells = self.cell_embeddings[cell_indices], len(cell_indices)
        return {"genes": self.gene_expression[idx], "cell_embeddings": cell_embs, "n_cells": n_cells,
                "label": self.labels[idx], "spot_idx": idx}


def collate_fn(batch):
    """Pad the variable number of nucleus tokens per spot and build the key mask."""
    genes = torch.stack([item["genes"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    n_cells = torch.LongTensor([item["n_cells"] for item in batch])
    max_cells = max(item["cell_embeddings"].shape[0] for item in batch)
    dim = batch[0]["cell_embeddings"].shape[1]
    padded = torch.zeros((len(batch), max_cells, dim))
    masks = torch.zeros((len(batch), max_cells), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["cell_embeddings"].shape[0]
        padded[i, :n] = item["cell_embeddings"]
        masks[i, :n] = True
    return {"genes": genes, "cell_embeddings": padded, "cell_masks": masks, "n_cells": n_cells, "labels": labels}


def make_loader(data, indices=None, batch_size=1024, shuffle=False, drop_last=False, labels_key="labels"):
    idx = np.arange(len(data["spot_to_cells"])) if indices is None else np.asarray(indices)
    dataset = SpatialMultimodalDataset(
        gene_expression=data["gene_expression"][idx],
        cell_embeddings=data["cell_embeddings"],
        spot_to_cells=[data["spot_to_cells"][i] for i in idx],
        labels=data[labels_key][idx],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                      num_workers=4, pin_memory=True, drop_last=drop_last)


def split_train_val(labels, val_fraction, seed):
    """9:1 split stratified by each spot's dominant cell type (classes with < 2 spots are split by hand)."""
    dominant = np.argmax(labels, axis=1)
    unique, counts = np.unique(dominant, return_counts=True)
    rare = set(unique[counts < 2])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    if rare:
        rare_mask = np.array([d in rare for d in dominant])
        common_idx, rare_idx = np.where(~rare_mask)[0], np.where(rare_mask)[0]
        c_train, c_val = next(sss.split(np.zeros(len(common_idx)), dominant[common_idx]))
        train_idx, val_idx = common_idx[c_train].tolist(), common_idx[c_val].tolist()
        np.random.shuffle(rare_idx)
        n_rare_val = max(1, int(val_fraction * len(rare_idx)))
        val_idx += rare_idx[:n_rare_val].tolist()
        train_idx += rare_idx[n_rare_val:].tolist()
        return train_idx, val_idx
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), dominant))
    return train_idx.tolist(), val_idx.tolist()


# =============================================================================
# Model
# =============================================================================

class GeneEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, cell_embeddings, mask=None):
        x = self.projection(cell_embeddings)
        query = self.query.expand(cell_embeddings.shape[0], -1, -1)
        pooled, _ = self.attention(query, x, x, key_padding_mask=(~mask if mask is not None else None))
        return pooled.squeeze(1)


class BranchedProportionDeconvModel(nn.Module):
    """Gene branch + morphology branch + fusion head, with a gene-only and a morphology-only auxiliary head."""

    def __init__(self, gene_input_dim, cell_embedding_dim, n_cell_types,
                 gene_hidden_dims=(512, 256), gene_output_dim=128, cell_hidden_dim=256, cell_num_heads=4,
                 fusion_hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        self.gene_encoder = GeneEncoder(gene_input_dim, list(gene_hidden_dims), gene_output_dim, dropout)
        self.cell_aggregator = AttentionPooling(cell_embedding_dim, cell_hidden_dim, cell_num_heads, dropout)
        layers, prev = [], gene_output_dim + cell_hidden_dim + 1
        for h in fusion_hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_cell_types))
        self.classifier = nn.Sequential(*layers)
        self.gene_head = nn.Sequential(nn.Linear(gene_output_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_cell_types))
        self.morph_head = nn.Sequential(nn.Linear(cell_hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_cell_types))

    def fused_features(self, genes, cell_embeddings, cell_masks, n_cells):
        gene_features = self.gene_encoder(genes)
        morph_features = self.cell_aggregator(cell_embeddings, cell_masks)
        n_cells_norm = (n_cells.float().unsqueeze(1) / 100.0).to(genes.device)
        return gene_features, morph_features, torch.cat([gene_features, morph_features, n_cells_norm], dim=1)

    def forward(self, genes, cell_embeddings, cell_masks, n_cells):
        """Fusion-head proportions only."""
        _, _, combined = self.fused_features(genes, cell_embeddings, cell_masks, n_cells)
        return F.softmax(self.classifier(combined), dim=-1)

    def forward_with_branches(self, genes, cell_embeddings, cell_masks, n_cells):
        """Fusion-head, gene-only and morphology-only proportions."""
        gene_features, morph_features, combined = self.fused_features(genes, cell_embeddings, cell_masks, n_cells)
        return (F.softmax(self.classifier(combined), dim=-1),
                F.softmax(self.gene_head(gene_features), dim=-1),
                F.softmax(self.morph_head(morph_features), dim=-1))


# =============================================================================
# Losses (torch) and metrics (numpy)
# =============================================================================

def pcc_loss(pred, target, eps=1e-8):
    """1 - mean over cell types of the Pearson correlation across the spots of the batch."""
    pred_c, tgt_c = pred - pred.mean(dim=0, keepdim=True), target - target.mean(dim=0, keepdim=True)
    num = (pred_c * tgt_c).sum(dim=0)
    den = torch.sqrt((pred_c ** 2).sum(dim=0) * (tgt_c ** 2).sum(dim=0) + eps)
    return 1.0 - (num / den).mean()


def rmse_loss(pred, target, eps=1e-12):
    """sqrt(mean over cell types of sum_i (T_ik - P_ik)^2 / S_k), S_k = sum_i T_ik; types absent from the batch are skipped."""
    S = target.sum(dim=0)
    valid = S > eps
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    mse_per_type = ((target[:, valid] - pred[:, valid]) ** 2).sum(dim=0) / (S[valid] + eps)
    return torch.sqrt(mse_per_type.mean() + eps)


def jsd_loss(pred, target, eps=1e-8):
    """Mean over cell types of the Jensen-Shannon divergence between the two distributions over spots."""
    S = target.sum(dim=0)
    valid = S > eps
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    t = target[:, valid] / (target[:, valid].sum(dim=0, keepdim=True) + eps)
    p = pred[:, valid] / (pred[:, valid].sum(dim=0, keepdim=True) + eps)
    m = 0.5 * (t + p)
    kl_t = (t * torch.log((t + eps) / (m + eps))).sum(dim=0)
    kl_p = (p * torch.log((p + eps) / (m + eps))).sum(dim=0)
    return (0.5 * kl_t + 0.5 * kl_p).mean()


class DeconvLoss(nn.Module):
    """w_pcc * (1 - PCC) + w_rmse * RMSE + w_jsd * JSD on proportions."""

    def __init__(self, w_pcc=1.0, w_rmse=1.0, w_jsd=1.0):
        super().__init__()
        self.w_pcc, self.w_rmse, self.w_jsd = w_pcc, w_rmse, w_jsd

    def forward(self, pred, target):
        l_pcc, l_rmse, l_jsd = pcc_loss(pred, target), rmse_loss(pred, target), jsd_loss(pred, target)
        return self.w_pcc * l_pcc + self.w_rmse * l_rmse + self.w_jsd * l_jsd, l_pcc, l_rmse, l_jsd


def compute_rmse_np(T, P):
    S = np.sum(T, axis=0)
    valid = S > 0
    return np.sqrt(np.mean(np.sum((T[:, valid] - P[:, valid]) ** 2, axis=0) / S[valid])) if valid.any() else np.nan


def compute_jsd_np(T, P, eps=1e-12):
    S = np.sum(T, axis=0)
    valid = S > 0
    if not valid.any():
        return np.nan
    T_v, P_v = T[:, valid], P[:, valid]
    vals = []
    for k in range(T_v.shape[1]):
        p, t = P_v[:, k] / (P_v[:, k].sum() + eps), T_v[:, k] / (T_v[:, k].sum() + eps)
        m = 0.5 * (p + t)
        vals.append(0.5 * np.sum((t + eps) * np.log((t + eps) / (m + eps))) + 0.5 * np.sum((p + eps) * np.log((p + eps) / (m + eps))))
    return np.mean(vals)


def compute_pcc_np(T, P):
    S = np.sum(T, axis=0)
    valid = S > 0
    if not valid.any():
        return np.nan
    T_v, P_v = T[:, valid], P[:, valid]
    vals = [np.corrcoef(T_v[:, k], P_v[:, k])[0, 1] if np.std(T_v[:, k]) > 0 and np.std(P_v[:, k]) > 0 else np.nan
            for k in range(T_v.shape[1])]
    return np.nanmean(vals)


def eval_proportions(pred_prop, gt_counts):
    """RMSE, JSD and PCC of predicted proportions against ground-truth counts (spots without cells are skipped).

    The validation score of the original script: with T, P of shape (types x spots), every metric is computed per
    spot across cell types and averaged over spots (PCC of the two composition vectors; RMSE = sqrt of the mean
    over spots of the summed squared error, i.e. sqrt(K) times the RMSE of the Methods; JSD of the two vectors).
    """
    valid = gt_counts.sum(axis=1) > 0
    gt_prop = gt_counts[valid] / gt_counts[valid].sum(axis=1, keepdims=True)
    pred_p = pred_prop[valid]
    return compute_rmse_np(gt_prop.T, pred_p.T), compute_jsd_np(gt_prop.T, pred_p.T), compute_pcc_np(gt_prop.T, pred_p.T)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        preds.append(model(batch["genes"].to(device), batch["cell_embeddings"].to(device),
                           batch["cell_masks"].to(device), batch["n_cells"].to(device)).cpu().numpy())
    return np.concatenate(preds)


def predict_target(model, target, batch_size=1024):
    """Proportions of the model on a target dict (pretrain.build_target) as a spots x cell types DataFrame."""
    device = next(model.parameters()).device
    return pd.DataFrame(predict(model, make_loader(target, batch_size=batch_size), device),
                        index=target["spot_ids"], columns=target["cell_types"])


# =============================================================================
# Training
# =============================================================================

def pretrain(source, hp, output_dir, monitor=None, device=None):
    """Stage-1 pretraining. Returns (model, checkpoint dict, history DataFrame) and saves output_dir/best_model.pt.

    source:  dict from build_source.  hp: hyperparameter dict (see the notebook).
    monitor: optional {"name": str, "data": dict from build_target} whose ground truth is evaluated every epoch
             for information only; model selection uses the validation split of the source only.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    if (output_dir / "best_model.pt").exists():                # cached run: delete the folder to retrain
        checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
        if (checkpoint["hyperparameters"] != dict(hp) or checkpoint["common_genes"] != source["common_genes"]
                or checkpoint["cell_types"] != source["cell_types"]):
            raise ValueError(f"{output_dir / 'best_model.pt'} was trained with other hyperparameters or data; "
                             "delete the folder to retrain")
        model = BranchedProportionDeconvModel(len(checkpoint["common_genes"]), source["cell_embeddings"].shape[1],
                                              len(checkpoint["cell_types"])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Using existing {output_dir / 'best_model.pt'} (validation score {checkpoint['best_val_score']:.4f})")
        return model, checkpoint, pd.read_csv(output_dir / "history.csv")
    np.random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = split_train_val(source["labels"], hp["val_fraction"], hp["seed"])
    print(f"Train: {len(train_idx):,} spots, validation: {len(val_idx):,} spots")
    train_loader = make_loader(source, train_idx, hp["batch_size"], shuffle=True, drop_last=True)
    val_loader = make_loader(source, val_idx, hp["batch_size"])
    monitor_loader = make_loader(monitor["data"], batch_size=hp["batch_size"]) if monitor else None

    model = BranchedProportionDeconvModel(
        gene_input_dim=len(source["common_genes"]), cell_embedding_dim=source["cell_embeddings"].shape[1],
        n_cell_types=len(source["cell_types"])).to(device)
    print(f"Model: {len(source['common_genes'])} genes, {len(source['cell_types'])} cell types, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    criterion = DeconvLoss(hp["w_pcc"], hp["w_rmse"], hp["w_jsd"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=hp["plateau_patience"], factor=hp["plateau_factor"], min_lr=hp["min_lr"])

    best_val_score, best_state, patience_counter, history = -float("inf"), copy.deepcopy(model.state_dict()), 0, []
    for epoch in range(hp["max_epochs"]):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            genes, cell_emb = batch["genes"].to(device), batch["cell_embeddings"].to(device)
            masks, n_cells, labels = batch["cell_masks"].to(device), batch["n_cells"].to(device), batch["labels"].to(device)
            main_pred, gene_pred, morph_pred = model.forward_with_branches(genes, cell_emb, masks, n_cells)
            loss_main, _, _, _ = criterion(main_pred, labels)
            loss_gene, _, _, _ = criterion(gene_pred, labels)
            loss_morph, _, _, _ = criterion(morph_pred, labels)
            loss = loss_main + hp["w_aux"] * (loss_gene + loss_morph)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            optimizer.step()
            epoch_loss += loss_main.item()
            n_batches += 1

        # validation: composite loss drives the LR schedule, the score PCC - RMSE - JSD drives model selection
        model.eval()
        val_loss, val_preds = 0.0, []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["genes"].to(device), batch["cell_embeddings"].to(device),
                             batch["cell_masks"].to(device), batch["n_cells"].to(device))
                val_loss += criterion(pred, batch["labels"].to(device))[0].item()
                val_preds.append(pred.cpu().numpy())
        val_loss /= len(val_loader)
        rmse_val, jsd_val, pcc_val = eval_proportions(np.concatenate(val_preds), source["raw_counts"][val_idx])
        val_score = pcc_val - rmse_val - jsd_val
        scheduler.step(val_loss)

        row = {"epoch": epoch + 1, "train_loss": epoch_loss / n_batches, "val_loss": val_loss,
               "val_pcc": pcc_val, "val_rmse": rmse_val, "val_jsd": jsd_val, "val_score": val_score,
               "lr": optimizer.param_groups[0]["lr"]}
        line = (f"Epoch {epoch + 1:4d} | loss={row['train_loss']:.4f} val_loss={val_loss:.4f} | "
                f"val PCC={pcc_val:.4f} RMSE={rmse_val:.4f} JSD={jsd_val:.4f} score={val_score:.4f} | lr={row['lr']:.1e}")
        if monitor:
            rmse_t, jsd_t, pcc_t = eval_proportions(predict(model, monitor_loader, device), monitor["data"]["raw_counts"])
            row.update({"monitor_pcc": pcc_t, "monitor_rmse": rmse_t, "monitor_jsd": jsd_t})
            line += f" | {monitor['name']}: PCC={pcc_t:.4f} RMSE={rmse_t:.4f} JSD={jsd_t:.4f}"
        history.append(row)

        if val_score > best_val_score:
            best_val_score, best_state, patience_counter = val_score, copy.deepcopy(model.state_dict()), 0
            line += "  <- best"
        else:
            patience_counter += 1
        print(line)
        if patience_counter >= hp["early_stop_patience"]:
            print(f"Early stopping after epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    checkpoint = {"model_state_dict": best_state, "cell_types": source["cell_types"],
                  "common_genes": source["common_genes"], "hyperparameters": dict(hp),
                  "best_val_score": best_val_score}
    history = pd.DataFrame(history)
    history.to_csv(output_dir / "history.csv", index=False)
    torch.save(checkpoint, output_dir / "best_model.pt")
    print(f"Best validation score {best_val_score:.4f}; checkpoint saved to {output_dir / 'best_model.pt'}")
    return model, checkpoint, history
