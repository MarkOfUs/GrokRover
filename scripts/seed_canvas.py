#!/usr/bin/env python3
"""Extract a real Jezero crater patch (USGS Mars2020 TRN mosaics) and compose
a seed canvas in the pipeline's 16:9 [texture | heightmap] format, ready for
Grok image_edit enhancement.

Data (co-registered, Eqc projection, same bounds):
  DTM   1 m/px  float32 elevation
  Ortho 25 cm/px uint8 HiRISE grayscale

Everything is read remotely via HTTP range requests (/vsicurl/) - the multi-GB
mosaics are never downloaded. Scouting samples one DTM row every ROW_STEP m
across the delta/landing region (~10 MB total), scores 200 m cells by relief,
and picks a medium-relief cell: interesting terrain, no canyon walls.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "20000000")

import rasterio
from rasterio.windows import Window

S3 = "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE"
DTM_URL = f"/vsicurl/{S3}/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif"
ORTHO_URL = f"/vsicurl/{S3}/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first.tif"

# scout region: box around the delta front / Perseverance landing area (proj meters)
SCOUT_X = (4583000.0, 4593000.0)
SCOUT_Y = (1091500.0, 1097500.0)
ROW_STEP = 25          # m between sampled scout rows
CELL = 200             # scout scoring cell size, m
PATCH_W, PATCH_H = 160, 180   # extracted patch size, m (matches 8:9 canvas half)
CANVAS_W, CANVAS_H = 2048, 1152

MARS_TINT = (1.00, 0.62, 0.42)  # rust multiply for grayscale HiRISE


def scout(dtm_bounds, nodata) -> tuple:
    """Sample rows across the scout box, score CELL-sized cells by relief,
    return (center_x, center_y) of the best medium-relief cell."""
    left, top = dtm_bounds.left, dtm_bounds.top
    col0, col1 = int(SCOUT_X[0] - left), int(SCOUT_X[1] - left)
    row0, row1 = int(top - SCOUT_Y[1]), int(top - SCOUT_Y[0])
    rows = list(range(row0, row1, ROW_STEP))

    def read_rows(chunk):
        with rasterio.open(DTM_URL) as ds:
            return [(r, ds.read(1, window=Window(col0, r, col1 - col0, 1))[0])
                    for r in chunk]

    chunks = [rows[i::8] for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        got = [rc for part in ex.map(read_rows, chunks) for rc in part]
    got.sort(key=lambda rc: rc[0])
    grid = np.stack([rc[1] for rc in got])            # (nrows, ncols) at 25x1 m
    grid[grid < -1e30] = np.nan

    cr, cc = CELL // ROW_STEP, CELL                    # cell shape in grid px
    best, best_score = None, -1.0
    scores = []
    for i in range(0, grid.shape[0] - cr, cr):
        for j in range(0, grid.shape[1] - cc, cc):
            cell = grid[i:i + cr, j:j + cc]
            if np.isnan(cell).any():
                continue
            relief = float(np.nanstd(cell))
            # slope proxy along sampled rows (1 m spacing in x)
            slope = float(np.abs(np.diff(cell, axis=1)).max())
            scores.append((relief, slope, i, j))
    if not scores:
        raise RuntimeError("no valid scout cells - adjust SCOUT_X/SCOUT_Y")
    # medium relief: drop the flattest half, reject cliff cells, take the
    # median of what remains so we get 'interesting, still driveable'
    scores.sort()
    ok = [s for s in scores[len(scores) // 2:] if s[1] < 8.0]
    relief, slope, i, j = ok[len(ok) // 2] if ok else scores[len(scores) // 2]
    cx = left + col0 + j + cc / 2
    cy = top - (row0 + i * ROW_STEP + (cr * ROW_STEP) / 2)
    print(f"scout: cell relief std {relief:.2f} m, max step {slope:.2f} m, "
          f"center ({cx:.0f}, {cy:.0f})")
    return cx, cy


def read_patch(url, bounds, cx, cy) -> np.ndarray:
    with rasterio.open(url) as ds:
        res = ds.res[0]
        w, h = int(PATCH_W / res), int(PATCH_H / res)
        col = int((cx - PATCH_W / 2 - ds.bounds.left) / res)
        row = int((ds.bounds.top - (cy + PATCH_H / 2)) / res)
        return ds.read(1, window=Window(col, row, w, h))


def normalize_height(dtm: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(dtm, [1, 99])
    return (np.clip((dtm - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def tint_ortho(gray: np.ndarray) -> Image.Image:
    lo, hi = np.percentile(gray[gray > 0], [2, 98])
    g = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    g = g ** 0.9
    rgb = np.stack([g * 255 * c for c in MARS_TINT], axis=-1)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent.parent / "worlds" / "jezero_real")
    ap.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"),
                    help="projected-meter center; skips scouting")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(DTM_URL) as ds:
        bounds, nodata = ds.bounds, ds.nodata
    cx, cy = args.center if args.center else scout(bounds, nodata)

    dtm = read_patch(DTM_URL, bounds, cx, cy)
    ortho = read_patch(ORTHO_URL, bounds, cx, cy)
    if (dtm < -1e30).any() or (ortho == 0).mean() > 0.02:
        raise RuntimeError("patch touches nodata - rerun or pass --center")

    np.save(args.out / "real_dtm.npy", dtm)
    hm = Image.fromarray(normalize_height(dtm))
    tex = tint_ortho(ortho)
    hm.save(args.out / "real_heightmap.png")
    tex.save(args.out / "real_ortho.png")

    half = (CANVAS_W // 2, CANVAS_H)
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H))
    canvas.paste(tex.resize(half, Image.BICUBIC), (0, 0))
    canvas.paste(hm.convert("RGB").resize(half, Image.BICUBIC), (CANVAS_W // 2, 0))
    canvas.save(args.out / "seed_canvas.png")
    rng = dtm.max() - dtm.min()
    print(f"patch {PATCH_W}x{PATCH_H} m at ({cx:.0f}, {cy:.0f}), "
          f"elev range {rng:.1f} m -> {args.out}/seed_canvas.png")


if __name__ == "__main__":
    main()
