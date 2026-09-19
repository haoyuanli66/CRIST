"""Step 2 of CRIST: segment the nuclei of the H&E image with CellViT and embed every nucleus.

CellViT (Hörst et al., Med Image Anal 2024, https://github.com/TIO-IKIM/CellViT, commit 40709a6)
is vendored in ./cellvit (code only; the CellViT-SAM-H checkpoint is downloaded separately, see
cellvit/README.md). It needs python 3.9.7 and torch >= 2.0 installed as described in that README
(cellvit/environment.yml + torch 2.0.0), so it runs in its own conda environment and this module
calls its two command-line entry points as subprocesses:

  1. cellvit/preprocessing/patch_extraction/main_extraction.py
       tiles the H&E image into 1024 px patches at 40x (0.2125 um/px) with 64 px overlap and
       Macenko stain normalisation -> <output_root>/<image name>/patches/
  2. cellvit/cell_segmentation/inference/cell_detection.py
       runs CellViT-SAM-H on every patch, removes duplicate detections in the overlap zones and
       writes <output_root>/<image name>/cell_detection/cells.pt

These are the settings with which the paper's cells.pt files were produced. cells.pt holds a
CellGraphDataWSI object with
  x          (n_nuclei, 1280)  embedding of every nucleus (mean of the ViT-H encoder tokens over
                               its bounding box; CRIST's morphology feature, D_m = 1280)
  positions  (n_nuclei, 2)     nucleus centroid in H&E pixels (x, y)
  contours   list of nucleus outlines in H&E pixels
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import yaml

CELLVIT_DIR = Path(__file__).resolve().parent / "cellvit"
CELLVIT_ENV = "cellvit_env"                                                   # name of the CellViT conda env ...
CELLVIT_PYTHON = Path(sys.prefix).parent / CELLVIT_ENV / "bin" / "python"    # ... assumed to sit next to the notebook's env
CELLVIT_CHECKPOINT = CELLVIT_DIR / "models/pretrained/CellViT/CellViT-SAM-H-x40.pth"   # download from the CellViT release
PATCH_SIZE = 1024              # px; cell_detection.py requires 1024
PATCH_OVERLAP_PERCENT = 6.25   # 6.25 % of 1024 px = 64 px, required by cell_detection.py
MAGNIFICATION = 40             # the CellViT-SAM-H-x40 checkpoint expects 40x patches


def _run(cmd):
    """Run a command in the CellViT environment, streaming its log into the notebook."""
    cmd = [str(c) for c in cmd]
    print("$ " + " ".join(cmd), flush=True)
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}   # the notebook's inline backend is unknown to other envs
    proc = subprocess.Popen(cmd, cwd=CELLVIT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**env, "MPLBACKEND": "Agg"})
    for line in proc.stdout:
        print(line.rstrip().split("\r")[-1], flush=True)   # keep only the last state of tqdm progress bars
    if proc.wait() != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {cmd[1]}")


def extract_patches(wsi_path, output_root, mpp):
    """Tile one H&E image into stain-normalised 1024 px patches at 40x. Returns the patched-slide folder."""
    wsi_path, output_root = Path(wsi_path), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "wsi_paths": str(wsi_path),
        "wsi_extension": wsi_path.suffix.lstrip("."),
        "output_path": str(output_root),          # <image name>/patches, plus CellViT's log and a copy of this config
        "patch_size": PATCH_SIZE,
        "patch_overlap": PATCH_OVERLAP_PERCENT,
        "target_mag": float(MAGNIFICATION),
        "normalize_stains": True,
        "min_intersection_ratio": 0.0,            # keep every tile, also pure background (as in the paper's runs)
        "processes": 8,
        "overwrite": False,                       # never True: CellViT would delete every other image's folder under output_root
        "hardware_selection": "openslide",        # the paper's runs read the images with OpenSlide
        "wsi_properties": {"slide_mpp": float(mpp), "magnification": MAGNIFICATION},   # plain .tif files carry no metadata
    }
    config_path = output_root / f"{wsi_path.stem}_patch_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    _run([CELLVIT_PYTHON, CELLVIT_DIR / "preprocessing/patch_extraction/main_extraction.py", "--config", config_path])
    return output_root / wsi_path.stem


def detect_cells(wsi_path, patched_dir, checkpoint, gpu=0, batch_size=2):
    """Run CellViT on the patches of one image (batch size 2 fits a 24 GB GPU). Returns the path of cells.pt."""
    _run([CELLVIT_PYTHON, CELLVIT_DIR / "cell_segmentation/inference/cell_detection.py",
          "--model", checkpoint, "--gpu", gpu, "--magnification", MAGNIFICATION, "--batch_size", batch_size,
          "process_wsi", "--wsi_path", wsi_path, "--patched_slide_path", patched_dir])
    return Path(patched_dir) / "cell_detection" / "cells.pt"


def segment_nuclei(wsi_path, output_root, mpp, checkpoint):
    """Patch extraction + CellViT inference for one H&E image; returns the path of cells.pt.

    Reuses <output_root>/<image name>/cell_detection/cells.pt when it already exists
    (delete that folder to recompute).
    """
    wsi_path, output_root = Path(wsi_path), Path(output_root)
    cells_pt = output_root / wsi_path.stem / "cell_detection" / "cells.pt"
    if cells_pt.exists():
        print(f"{wsi_path.name}: using existing {cells_pt}")
        return cells_pt
    patched_dir = extract_patches(wsi_path, output_root, mpp)
    return detect_cells(wsi_path, patched_dir, checkpoint)


def read_region(tif_path, x0, y0, width, height):
    """Read a (height, width, 3) crop of the full-resolution H&E image by decoding only the tiles it covers."""
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        tw, th = page.tilewidth, page.tilelength
        n_tiles_x = -(-page.imagewidth // tw)
        jpegtables, decode = page.jpegtables, page.decode      # resolve these first: building them moves the file pointer
        out = np.zeros((height, width, 3), dtype=np.uint8)
        for ty in range(y0 // th, (y0 + height - 1) // th + 1):
            for tx in range(x0 // tw, (x0 + width - 1) // tw + 1):
                index = ty * n_tiles_x + tx
                tif.filehandle.seek(page.dataoffsets[index])
                data = tif.filehandle.read(page.databytecounts[index])
                tile, _, shape = decode(data, index, jpegtables=jpegtables)
                tile = tile.reshape(shape[-3:])
                ys, xs = ty * th, tx * tw
                sy0, sx0, sy1, sx1 = max(y0, ys), max(x0, xs), min(y0 + height, ys + th), min(x0 + width, xs + tw)
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = tile[sy0 - ys:sy1 - ys, sx0 - xs:sx1 - xs, :3]
    return out


def read_overview(tif_path, level=4):
    """Read one level of the image pyramid (level 4 = 1/16 of the full resolution) and its downsampling factor."""
    with tifffile.TiffFile(tif_path) as tif:
        series = tif.series[0]
        image = series.levels[level].asarray()
        factor = series.levels[0].shape[1] / image.shape[1]
    return image[..., :3], factor


def load_cells(cells_pt):
    """Load cells.pt -> dict(embeddings (n, 1280), positions (n, 2) in H&E px, contours)."""
    if str(CELLVIT_DIR) not in sys.path:
        sys.path.insert(0, str(CELLVIT_DIR))     # cells.pt pickles CellViT's CellGraphDataWSI class
    graph = torch.load(cells_pt, map_location="cpu", weights_only=False)
    return {"embeddings": graph.x, "positions": graph.positions, "contours": graph.contours}
