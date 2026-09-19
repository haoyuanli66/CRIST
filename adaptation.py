"""Step 5 of CRIST: Stage-2 co-teaching adaptation to the target sample with Cell2location pseudo-labels.

Two peers f and g start from the Stage-1 checkpoint; g's trainable weights are perturbed by Gaussian noise
(std = perturbation_scale x the tensor's own std, seed_g). The gene encoder, the morphology encoder and the
morphology head are frozen; only the fusion head and the gene head are updated.
In every minibatch each peer scores every spot by
    s_i = MSE(fusion prediction, pseudo-label) + gamma * MSE(morphology-head prediction, pseudo-label)
(the frozen morphology head is an H&E-based anchor that ambient-RNA noise in the pseudo-labels cannot
affect), keeps the fraction rho_e = 1 - tau * min((e + 1) / E_tau, 1) of spots with the smallest score and
its peer is updated on that selection with the Stage-1 loss w_pcc (1 - PCC) + w_rmse RMSE + w_jsd JSD.
After the last epoch the peer whose fusion-head loss against the pseudo-labels over all target spots is
lower provides the transferred composition (as described in the paper).
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evaluation import normalize_celltype_name
from pretrain import (BranchedProportionDeconvModel, DeconvLoss, eval_proportions, jsd_loss, make_loader,
                      pcc_loss, predict, rmse_loss)


def pseudo_label_proportions(pseudo_df, spot_ids, cell_types):
    """Cell2location abundances -> (spots x cell types) proportions in the model's cell-type order."""
    lookup = {normalize_celltype_name(c): c for c in pseudo_df.columns}
    aligned = np.zeros((len(spot_ids), len(cell_types)), dtype=np.float32)
    rows = pseudo_df.reindex(spot_ids)
    for i, ct in enumerate(cell_types):
        key = normalize_celltype_name(ct)
        if key in lookup:
            aligned[:, i] = np.nan_to_num(rows[lookup[key]].to_numpy(dtype=np.float32))
    aligned = np.clip(aligned, 0, None)
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    return aligned / row_sums


def freeze_encoders_and_morph(model):
    for module in (model.gene_encoder, model.cell_aggregator, model.morph_head):
        for param in module.parameters():
            param.requires_grad = False
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  frozen {n_total - n_trainable:,} parameters (gene encoder, morphology encoder, morphology head); "
          f"trainable {n_trainable:,} (fusion head, gene head)")


def forget_rate(epoch, tau, Tk):
    """Fraction of spots rejected in `epoch` (0-based): tau * min((epoch + 1) / Tk, 1)."""
    return tau * min((epoch + 1) / Tk, 1.0)


def per_sample_score(model, batch, device, gamma):
    """Reliability score of every spot of the batch (lower = cleaner), the fusion prediction and the labels."""
    genes, cell_emb = batch["genes"].to(device), batch["cell_embeddings"].to(device)
    masks, n_cells, labels = batch["cell_masks"].to(device), batch["n_cells"].to(device), batch["labels"].to(device)
    main_pred, _, morph_pred = model.forward_with_branches(genes, cell_emb, masks, n_cells)
    mse_main = ((main_pred - labels) ** 2).mean(dim=1)
    mse_morph = ((morph_pred.detach() - labels) ** 2).mean(dim=1)
    return mse_main + gamma * mse_morph, main_pred, labels


def coteaching_step(model_f, model_g, opt_f, opt_g, batch, device, keep_rate, hp):
    """One co-teaching update: each peer is trained on the spots its partner scored as most reliable."""
    model_f.train()
    model_g.train()
    score_f, pred_f, labels = per_sample_score(model_f, batch, device, hp["gamma"])
    score_g, _, _ = per_sample_score(model_g, batch, device, hp["gamma"])
    n_keep = max(1, int(score_f.shape[0] * keep_rate))
    sel_f = torch.sort(score_f.detach())[1][:n_keep]
    sel_g = torch.sort(score_g.detach())[1][:n_keep]

    def loss_on(pred, lab):
        return hp["w_pcc"] * pcc_loss(pred, lab) + hp["w_rmse"] * rmse_loss(pred, lab) + hp["w_jsd"] * jsd_loss(pred, lab)

    opt_f.zero_grad()                                   # f learns from g's selection
    loss_f = loss_on(pred_f[sel_g], labels[sel_g])
    loss_f.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model_f.parameters() if p.requires_grad], hp["grad_clip"])
    opt_f.step()

    _, pred_g, labels_g = per_sample_score(model_g, batch, device, hp["gamma"])   # g learns from f's selection
    opt_g.zero_grad()
    loss_g = loss_on(pred_g[sel_f], labels_g[sel_f])
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model_g.parameters() if p.requires_grad], hp["grad_clip"])
    opt_g.step()
    return loss_f.item(), loss_g.item()


def fusion_loss_on_target(model, loader, criterion, device):
    """Stage-1 composite loss of the fusion head against the pseudo-labels, over all target spots at once."""
    preds = torch.from_numpy(predict(model, loader, device))
    labels = loader.dataset.labels
    return criterion(preds, labels)[0].item()


def adapt(checkpoint, target, pseudo_df, hp, output_dir, device=None):
    """Co-teaching adaptation. Returns (predictions DataFrame, history DataFrame, name of the selected peer).

    target: dict from pretrain.build_target (its labels, if any, are used for monitoring only).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    cell_types, genes = checkpoint["cell_types"], checkpoint["common_genes"]
    if (output_dir / "adapted_model.pt").exists():             # cached run: delete the folder to redo the adaptation
        saved = torch.load(output_dir / "adapted_model.pt", map_location="cpu", weights_only=False)
        if saved["hyperparameters"] != dict(hp) or saved["cell_types"] != cell_types or saved["common_genes"] != genes:
            raise ValueError(f"{output_dir / 'adapted_model.pt'} comes from other hyperparameters or data; delete the folder to redo")
        print(f"Using existing {output_dir / 'crist_proportions.csv'} (peer {saved['peer']})")
        return (pd.read_csv(output_dir / "crist_proportions.csv", index_col=0), pd.read_csv(output_dir / "history.csv"), saved["peer"])
    output_dir.mkdir(parents=True, exist_ok=True)
    target = dict(target)
    target["pseudo_labels"] = pseudo_label_proportions(pseudo_df, target["spot_ids"], cell_types)
    train_loader = make_loader(target, batch_size=hp["batch_size"], shuffle=True, labels_key="pseudo_labels")
    pseudo_loader = make_loader(target, batch_size=hp["batch_size"], labels_key="pseudo_labels")
    has_gt = target["raw_counts"].sum() > 0

    peers = {}
    for name in ("f", "g"):
        model = BranchedProportionDeconvModel(len(genes), target["cell_embeddings"].shape[1], len(cell_types)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"peer {name}:")
        freeze_encoders_and_morph(model)
        peers[name] = model
    torch.manual_seed(hp["seed_g"])                     # seeds the perturbation of g (and the batch order)
    with torch.no_grad():
        for p_f, p_g in zip(peers["f"].parameters(), peers["g"].parameters()):
            if p_g.requires_grad:
                p_g.add_(torch.randn_like(p_g) * p_g.std() * hp["perturbation_scale"])
    opts = {name: torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=hp["weight_decay"])
            for name, m in peers.items()}
    criterion = DeconvLoss(hp["w_pcc"], hp["w_rmse"], hp["w_jsd"])

    if has_gt:                                              # information only, as in the original script
        rmse, jsd, pcc = eval_proportions(predict(peers["f"], pseudo_loader, device), target["raw_counts"])
        print(f"Before adaptation | vs ground truth: PCC={pcc:.4f} RMSE={rmse:.4f} JSD={jsd:.4f}")
    history = []
    for epoch in range(hp["epochs"]):
        keep_rate = 1.0 - forget_rate(epoch, hp["tau"], hp["warmup_epochs"])
        sums, n_batches = np.zeros(2), 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            sums += coteaching_step(peers["f"], peers["g"], opts["f"], opts["g"], batch, device, keep_rate, hp)
            n_batches += 1
        row = {"epoch": epoch + 1, "keep_rate": keep_rate, "loss_f": sums[0] / n_batches, "loss_g": sums[1] / n_batches}
        line = f"Epoch {epoch + 1:3d} | keep={keep_rate:.2f} | loss f={row['loss_f']:.4f} g={row['loss_g']:.4f}"
        if has_gt:
            for name, model in peers.items():
                rmse, jsd, pcc = eval_proportions(predict(model, pseudo_loader, device), target["raw_counts"])
                row.update({f"pcc_{name}": pcc, f"rmse_{name}": rmse, f"jsd_{name}": jsd})
                line += f" | {name} vs ground truth: PCC={pcc:.4f} RMSE={rmse:.4f} JSD={jsd:.4f}"
        history.append(row)
        print(line)

    pseudo_loss = {name: fusion_loss_on_target(model, pseudo_loader, criterion, device) for name, model in peers.items()}
    selected = min(pseudo_loss, key=pseudo_loss.get)
    print(f"Fusion-head loss against the pseudo-labels after the last epoch: f={pseudo_loss['f']:.4f}, "
          f"g={pseudo_loss['g']:.4f} -> peer {selected} provides the transferred composition")
    predictions = pd.DataFrame(predict(peers[selected], pseudo_loader, device), index=target["spot_ids"], columns=cell_types)
    predictions.to_csv(output_dir / "crist_proportions.csv")
    history = pd.DataFrame(history)
    history.to_csv(output_dir / "history.csv", index=False)
    torch.save({"model_state_dict": peers[selected].state_dict(), "cell_types": cell_types, "common_genes": genes,
                "hyperparameters": dict(hp), "peer": selected}, output_dir / "adapted_model.pt")
    return predictions, history, selected
