#!/usr/bin/env python3
"""Batch-generate REAL Mars worlds: random Jezero patches -> worlds/<slug>/.

v2 counterpart of generate_worlds.py (self-contained on purpose - that file
is under active edit). Per world:

  1. random valid 160x180 m patch from the USGS Mars2020 TRN mosaics
     (DTM 1 m/px + HiRISE ortho 25 cm/px, ranged HTTP reads, no downloads)
  2. seed canvas [tinted ortho | DTM grayscale], 2048x1152
  3. one grok image_edit in the v1 canvas language (art mode) + invented TITLE
  4. gates: heightmap detail ratio >= 0.30 AND corr(height panel, real DTM)
     >= 0.45 (reject photo-copies); up to 3 attempts
  5. worlds/jezero-<slug>/  scene.usda + assets/ (grok_originals keeps the
     returned canvas, the real-data seed, real_dtm.npy, meta.json)
  6. renders/<slug>-hero.png + <slug>-overhead.png

    generate_real_worlds.py --count 10 [--jobs 3]
"""

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from seed_canvas import (DTM_URL, ORTHO_URL, SCOUT_X, SCOUT_Y, ROW_STEP, CELL,
                         PATCH_W, PATCH_H, CANVAS_W, CANVAS_H,
                         read_patch, normalize_height, tint_ortho)  # noqa: E402
from generate_assets import call_grok_full, split_canvas  # noqa: E402
from build_scene import build as build_usda  # noqa: E402

import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

ROOT = Path(__file__).parent.parent
MARS_R = 3396190.0  # m, Eqc latTs0 sphere
SEED = 7            # reproducible patch draw

EDIT_INSTRUCTION = (
    "use your image edit tool on the image file at {path}. it is real mars "
    "orbital data of jezero crater: one 16:9 canvas split 50/50 side by side, "
    "left half the terrain picture, right half its matching grayscale "
    "elevation heightmap. redraw this same canvas as a high-detail mars "
    "terrain map: keep the layout, landforms and slopes of the input exactly "
    "in place on both halves, adding realistic rover-scale rocks, boulders "
    "and regolith. the right half must be matte pure elevation data (white = "
    "high, black = low), no shading, no lighting, no albedo, as crisp as the "
    "left picture and matching it precisely. then end your reply with exactly "
    "two lines:\n"
    "TITLE: <a short evocative name you invent for this terrain>\n"
    "PATH: <the absolute file path of the saved image>"
)


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:48] or "world"


def hm_detail_ratio(canvas_path: Path) -> float:
    img = Image.open(canvas_path).convert("RGB")
    w, h = img.size

    def energy(im):
        return float(np.asarray(
            im.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32).std())

    return energy(img.crop((w // 2, 0, w, h))) / max(energy(img.crop((0, 0, w // 2, h))), 1e-6)


def dtm_corr(canvas_path: Path, dtm: np.ndarray) -> float:
    """Correlation of the canvas' height half against the real DTM."""
    img = Image.open(canvas_path).convert("RGB")
    w, h = img.size
    panel = np.asarray(img.crop((w // 2, 0, w, h)).convert("L")
                       .resize((256, 288), Image.BICUBIC), dtype=np.float32)
    ref = np.asarray(Image.fromarray(normalize_height(dtm))
                     .resize((256, 288), Image.BICUBIC), dtype=np.float32)
    return float(np.corrcoef(panel.ravel(), ref.ravel())[0, 1])


def candidate_cells() -> list:
    """All valid (medium-relief, non-cliff) scout cells as center coords."""
    with rasterio.open(DTM_URL) as ds:
        left, top = ds.bounds.left, ds.bounds.top
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
    grid = np.stack([rc[1] for rc in got])
    grid[grid < -1e30] = np.nan

    cr, cc = CELL // ROW_STEP, CELL
    out = []
    reliefs = []
    for i in range(0, grid.shape[0] - cr, cr):
        for j in range(0, grid.shape[1] - cc, cc):
            cell = grid[i:i + cr, j:j + cc]
            if np.isnan(cell).any():
                continue
            relief = float(np.nanstd(cell))
            slope = float(np.abs(np.diff(cell, axis=1)).max())
            reliefs.append(relief)
            out.append((relief, slope, i, j))
    med = float(np.median(reliefs))
    cells = []
    for relief, slope, i, j in out:
        if relief >= med and slope < 8.0:
            cx = left + col0 + j + cc / 2
            cy = top - (row0 + i * ROW_STEP + (cr * ROW_STEP) / 2)
            cells.append((cx, cy, relief))
    return cells


def make_world(idx: int, cx: float, cy: float, relief: float) -> str:
    tmp = ROOT / "worlds" / f"_real_tmp_{idx}"
    tmp.mkdir(parents=True, exist_ok=True)
    with rasterio.open(DTM_URL) as ds:
        bounds = ds.bounds
    dtm = read_patch(DTM_URL, bounds, cx, cy)
    ortho = read_patch(ORTHO_URL, bounds, cx, cy)
    if (dtm < -1e30).any() or (ortho == 0).mean() > 0.02:
        shutil.rmtree(tmp)
        return f"[{idx}] SKIP nodata at ({cx:.0f},{cy:.0f})"

    np.save(tmp / "real_dtm.npy", dtm)
    hm = Image.fromarray(normalize_height(dtm))
    tex = tint_ortho(ortho)
    half = (CANVAS_W // 2, CANVAS_H)
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H))
    canvas.paste(tex.resize(half, Image.BICUBIC), (0, 0))
    canvas.paste(hm.convert("RGB").resize(half, Image.BICUBIC), (CANVAS_W // 2, 0))
    seed_path = tmp / "seed_canvas.png"
    canvas.save(seed_path)

    title, result = None, None
    for attempt in range(3):
        try:
            img_path, out_text = call_grok_full(
                EDIT_INSTRUCTION.format(path=seed_path.resolve()))
            ratio = hm_detail_ratio(img_path)
            corr = dtm_corr(img_path, dtm)
            if ratio < 0.30 or corr < 0.45:
                print(f"[{idx}] attempt {attempt}: gate fail "
                      f"(detail {ratio:.2f}, corr {corr:.2f})", flush=True)
                continue
            m = re.search(r"TITLE:\s*(.+)", out_text)
            title = m.group(1).strip() if m else f"patch {idx}"
            result = (img_path, ratio, corr)
            break
        except Exception as e:
            print(f"[{idx}] attempt {attempt} error: {e}", flush=True)
    if result is None:
        shutil.rmtree(tmp)
        return f"[{idx}] FAIL all attempts at ({cx:.0f},{cy:.0f})"

    img_path, ratio, corr = result
    slug = "jezero-" + slugify(title)
    w = ROOT / "worlds" / slug
    if w.exists():
        n = 2
        while (ROOT / "worlds" / f"{slug}-{n}").exists():
            n += 1
        slug = f"{slug}-{n}"
        w = ROOT / "worlds" / slug
    (w / "assets" / "grok_originals").mkdir(parents=True, exist_ok=True)

    Image.open(img_path).convert("RGB").save(w / "assets/grok_originals/canvas.png")
    shutil.copy2(seed_path, w / "assets/grok_originals/seed_canvas_realdata.png")
    shutil.copy2(tmp / "real_dtm.npy", w / "assets/grok_originals/real_dtm.npy")
    split_canvas(w / "assets/grok_originals/canvas.png", w / "assets")
    shutil.copy2(ROOT / "assets/MarsSky.png", w / "assets/MarsSky.png")
    meta = {
        "source": "USGS Mars2020 TRN HiRISE mosaics (DTM 1 m/px, ortho 25 cm/px)",
        "center_proj_m": [cx, cy],
        "lon_lat_deg": [cx / MARS_R * 180 / np.pi, cy / MARS_R * 180 / np.pi],
        "patch_m": [PATCH_W, PATCH_H],
        "elev_range_m": float(dtm.max() - dtm.min()),
        "relief_std_m": relief,
        "gates": {"hm_detail_ratio": ratio, "dtm_corr": corr},
        "title": title,
    }
    (w / "assets/grok_originals/meta.json").write_text(json.dumps(meta, indent=2))
    build_usda(w / "assets", w / "scene.usda")
    for cam, name in [("HeroCam", "hero"), ("OverheadCam", "overhead")]:
        subprocess.run(
            ["usdrecord", "--camera", f"/World/Cameras/{cam}", "--imageWidth",
             "1600", "scene.usda", str(ROOT / "renders" / f"{slug}-{name}.png")],
            cwd=w, capture_output=True, timeout=600)
    shutil.rmtree(tmp)
    return (f"[{idx}] OK {slug} ({relief:.1f} m relief, detail {ratio:.2f}, "
            f"corr {corr:+.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N cells of the draw (continue a batch)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="shuffle seed; change it to draw cells in a new order")
    args = ap.parse_args()

    print("scouting candidate cells ...", flush=True)
    cells = candidate_cells()
    print(f"{len(cells)} valid cells; drawing {args.count} (seed {args.seed})",
          flush=True)
    random.Random(args.seed).shuffle(cells)

    # never redraw ground an existing world already covers
    taken = set()
    for p in (ROOT / "worlds").glob("*/assets/grok_originals/meta.json"):
        try:
            c = json.loads(p.read_text()).get("center_proj_m")
            if c:
                taken.add((float(c[0]), float(c[1])))
        except Exception:
            pass
    fresh = [c for c in cells if (float(c[0]), float(c[1])) not in taken]
    print(f"{len(taken)} cells already covered by existing worlds; "
          f"{len(fresh)} fresh", flush=True)
    picked = fresh[args.skip:args.skip + args.count]

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for line in ex.map(lambda t: make_world(*t),
                           [(i, cx, cy, r) for i, (cx, cy, r) in enumerate(picked)]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
