# CRIST

Reproducible demonstration of CRIST, a cross-resolution integration framework that transfers cell-type composition
from annotated high-resolution spatial transcriptomics (10x Xenium / Visium HD) to low-resolution spots, on the
human skin setting of the paper: pretraining on two healthy Xenium sections, adaptation to an independent cutaneous
melanoma section, and comparison with Cell2location against the section's cell-level ground truth.

Everything is driven by one notebook, `demo.ipynb`, which runs top to bottom in about 50 minutes on one GPU
(24 GB) and reproduces the numbers of the paper's skin experiment (Cell2location PCC 0.73 → CRIST 0.88 on the
melanoma section). Each step sets its hyperparameters in the cell before it runs, with the values of the paper's
Supplementary Note.

## Steps

| step | what it does | module |
|---|---|---|
| 0 | cell-type labels for every cell of the high-resolution samples with TACCO and the single-cell atlas (Xenium; Visium HD with segmentation; Visium HD without segmentation via CellViT nuclei + bin2cell), then a manual correction of classes that cannot occur in a sample | `annotation.py` |
| 1 | aggregation of each section into Visium-sized pseudo-spots on a 55 µm hexagonal lattice: transcript counts plus the exact cell composition of every spot | `pseudo_spots.py` |
| 2 | nucleus segmentation of the registered H&E image with CellViT-SAM-H: centroid, outline and 1,280-d embedding per nucleus | `nuclei_segmentation.py`, `cellvit/` |
| 3 | Stage 1: composition-guided multimodal pretraining (gene branch, attention-pooled morphology branch, fusion head, auxiliary heads) on the pseudo-spots of the pretraining sections | `pretrain.py` |
| 4 | pseudo-labels for the target section with Cell2location and the atlas | `pseudo_labels.py` |
| 5 | Stage 2: morphology-informed co-teaching adaptation of the pretrained model to the target with the pseudo-labels; comparison with the model before adaptation | `adaptation.py` |
| 6 | injection of the transferred composition into Cell2location's abundance prior | `prior_injection.py`, `cell2location_presence_module.py` |
| 7 | per-spot PCC / RMSE / JSD against the ground truth, significance test, box plots, composition maps and improvement maps | `evaluation.py`, `plotting.py` |

## Folder layout

```
demo.ipynb                        the demonstration (executed; outputs stored)
*.py                              one small module per step (see the table above)
cellvit/                          CellViT source code (upstream commit 40709a6); the checkpoint goes to
                                  cellvit/models/pretrained/CellViT/CellViT-SAM-H-x40.pth
bin2cell/                         bin2cell 0.3.3 package (used by step 0 for Visium HD without segmentation)
envs/                             conda environment files (see below)
data/<sample>/                    raw inputs of each Xenium section: cell_feature_matrix.h5, cells.parquet,
                                  transcripts.parquet, cell_boundaries.csv, <sample>.tif (H&E registered to the Xenium frame)
data/human_skin_normal_plus_melanoma/sc.h5ad
                                  the 13-class single-cell skin atlas (step 0 labels, step 4 reference)
cellvit_output/<sample>/cell_detection/cells.pt
                                  CellViT nuclei of each section; step 2 reuses them instead of running CellViT (~13-30 min per section)
output/                           everything the notebook computes; every step reuses its output folder if it exists
                                  (delete a folder to recompute that step)
MANIFEST.tsv                      md5, size and path of every data file
```

Samples: `Xenium_human_skin_s1`, `Xenium_human_skin_s2` (healthy skin, pretraining) and `Xenium_skin_disease`
(cutaneous melanoma, target). All paths in the notebook are relative to this folder.

## Setup

1. **Data.** Download the data archive (link provided with the submission) and unpack it here so that the
   `data/` and `cellvit_output/` folders above exist (about 5.2 GB); verify the files with
   `awk 'NR>1{print $1"  "$3}' MANIFEST.tsv | md5sum -c`.
   Download the CellViT-SAM-H (40×) checkpoint from the CellViT release
   (https://drive.google.com/uc?export=download&id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV, 2.8 GB) to
   `cellvit/models/pretrained/CellViT/CellViT-SAM-H-x40.pth`. It is only needed if `cellvit_output/` is absent.

2. **Environments.** Four conda environments; the kernel env runs the notebook, the other three are called as
   subprocesses and must be created with exactly these names (or edit `TACCO_ENV` / `BIN2CELL_ENV` in
   `annotation.py` and `CELLVIT_ENV` in `nuclei_segmentation.py`). They are expected next to the kernel env in the
   same conda `envs/` directory.

   ```bash
   conda env create -f envs/crist.yml        # kernel: torch 2.6 (CUDA), scanpy, cell2location 0.1.4, scvi-tools 1.3.0, jupyter
   conda env create -f envs/tacco.yml        # tacco_env: TACCO 0.4.0                       (step 0)
   conda env create -f envs/bin2cell.yml     # visium_hd_annotation: stardist, opencv, ... (step 0, Visium HD without segmentation only)
   conda env create -f envs/cellvit.yml      # cellvit_env: python 3.9.7, torch 2.0         (step 2, only if cellvit_output/ is absent)
   ```

   `envs/exported/` holds the exact package lists of the environments the notebook was executed in.

3. **Run.**

   ```bash
   conda activate crist
   jupyter lab demo.ipynb            # run all cells, top to bottom
   ```

   or without the browser: `jupyter nbconvert --to notebook --execute --inplace demo.ipynb`.

Hardware used: one NVIDIA RTX 4090 (24 GB), 32 CPU cores, 128 GB RAM (step 0 on a whole Visium HD section with
bin2cell needs about 50 GB; the skin demo needs far less).

## Outputs

`output/annotation/` cell labels, `output/pretrain/best_model.pt` the Stage-1 checkpoint, `output/cell2location/`
the Cell2location reference signatures, pseudo-labels and CRIST-injected result, `output/adaptation/` the adapted
model and its proportions, `output/crist_vs_cell2location_boxplots.png` the benchmark-style figure.
