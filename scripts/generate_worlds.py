#!/usr/bin/env python3
"""Batch-generate titled Mars worlds into worlds/<slug>/.

Each world is one Grok canvas generation (terrain | heightmap split, jointly
generated so they stay aligned). Grok also invents a short title for the
terrain; the slug becomes the directory name. Layout per world (self-contained,
drag into Isaac Sim as a unit):

    worlds/<slug>/
        scene.usda
        assets/
            grok_originals/canvas.<ext>   # untouched split file (dev post)
            terrain_texture.png
            heightmap.png
            MarsSky.png                   # shared sky, copied in

The sky is generated once (assets/MarsSky.png at the project root) and reused.

    generate_worlds.py --count 8 [--jobs 4]
"""

import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from generate_assets import (INSTRUCTIONS, api_generate, author_canvas_prompt,  # noqa: E402
                             call_grok, keep_original, split_canvas, upscale)
from build_scene import build as build_usda  # noqa: E402

ROOT = Path(__file__).parent.parent
MIN_HM_DETAIL = 0.30  # heightmap/terrain edge-energy ratio gate (bad run2 = 0.245)

WORLD_INSTRUCTION = (
    "make a mars terrain map, should be a cool topdown 2d view. generate it as "
    "one 16:9 canvas split 50/50 side by side: left half the terrain picture, "
    "right half its matching grayscale elevation heightmap, aligned. the "
    "heightmap must be matte pure elevation data (white = high, black = low) "
    "with no shading or lighting baked in. both halves must be rendered at "
    "the same resolution and the same level of fine detail - the heightmap "
    "as crisp as the terrain picture, its features matching precisely. "
    "use your image generation tool only; do not create, copy, or modify any "
    "files on disk. then end your reply with exactly two lines:\n"
    "TITLE: <a short evocative name you invent for this terrain>\n"
    "PATH: <the absolute file path of the saved image>"
)


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:48] or "world"


def ensure_sky() -> Path:
    sky = ROOT / "assets" / "MarsSky.png"
    if not sky.exists():
        print("generating shared MarsSky.png ...")
        sky.parent.mkdir(parents=True, exist_ok=True)
        from generate_assets import sky as gen_sky
        gen_sky(sky.parent, 0)
    return sky


def hm_detail_ratio(canvas_path: Path) -> float:
    """Edge-energy of heightmap half relative to terrain half (1.0 = equal)."""
    img = Image.open(canvas_path).convert("RGB")
    w, h = img.size

    def energy(im):
        return float(np.asarray(
            im.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32).std())

    return energy(img.crop((w // 2, 0, w, h))) / max(energy(img.crop((0, 0, w // 2, h))), 1e-6)


def gen_world(idx: int, out_root: Path, sky: Path, levels: int) -> str:
    try:
        # step 1: skill-loaded grok agent authors the Imagine prompt + title
        prompt, title = author_canvas_prompt()
        # step 2: we fire the generation at the API with resolution enforced
        tmp_dir = out_root / ".tmp"
        tmp_dir.mkdir(exist_ok=True)
        best = None  # (ratio, src)
        for attempt in range(2):
            src = api_generate(prompt, tmp_dir / f"canvas_{idx}_{attempt}")
            ratio = hm_detail_ratio(src)
            if best is None or ratio > best[0]:
                best = (ratio, src)
            if ratio >= MIN_HM_DETAIL:
                break
            print(f"[{idx}] heightmap too soft (ratio {ratio:.2f}), retrying")
        ratio, src = best
        slug = slugify(title)
        wdir = out_root / slug
        n = 2
        while wdir.exists():
            wdir = out_root / f"{slug}-{n}"
            n += 1
        assets = wdir / "assets"
        assets.mkdir(parents=True)
        keep_original(src, assets, "canvas")
        split_canvas(src, assets)
        shutil.copy2(sky, assets / "MarsSky.png")
        if levels > 0:
            try:
                upscale(assets, levels)
                shutil.rmtree(assets / "upscale_work", ignore_errors=True)
            except Exception as e:
                print(f"[{idx}] upscale failed, using base res: {e}", file=sys.stderr)
        build_usda(assets, wdir / "scene.usda")
        check = subprocess.run(["usdchecker", str(wdir / "scene.usda")],
                               capture_output=True, text=True)
        status = "PASS" if check.returncode == 0 else "FAIL usdchecker"
        soft = "" if ratio >= MIN_HM_DETAIL else f"  [soft heightmap {ratio:.2f}]"
        print(f"[{idx}] {status}  {wdir.name}  ({title}){soft}")
        return f"{status} {wdir.name}"
    except Exception as e:
        print(f"[{idx}] FAIL: {e}", file=sys.stderr)
        return f"FAIL world {idx}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--levels", type=int, default=3,
                    help="recursive upscale levels per world (0 = skip)")
    ap.add_argument("--out", type=Path, default=ROOT / "worlds")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sky = ensure_sky()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda i: gen_world(i, args.out, sky, args.levels),
                              range(1, args.count + 1)))
    ok = sum(1 for r in results if r.startswith("PASS"))
    print(f"\n{ok}/{len(results)} worlds built OK in {args.out}")


if __name__ == "__main__":
    main()
