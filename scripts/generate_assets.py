#!/usr/bin/env python3
"""Generate Grok Imagine assets: terrain canvas, sky, and recursive upscale.

Each generation is one headless `grok -p` call; Grok's own skills do the
prompt-craft. Untouched originals are kept in assets/grok_originals/.

  --mode canvas            terrain picture + matte heightmap, one 16:9 canvas
  --mode sky               mars sky texture (dome light)
  --mode upscale --levels N
        Recursive quadrant re-rendering of terrain_texture.png ONLY (the
        heightmap is never touched, so the mesh stays identical). Each level
        splits every tile into 4 overlapping quadrants and has Grok image_edit
        re-render each at full output res -> 2x linear resolution per level.
        Results are color-matched to their source crop and feather-stitched.
        N levels => 2^N x resolution, 4 + 16 + ... + 4^N edit calls.
        Resumable: finished tiles are cached in assets/upscale_work/.
"""

import argparse
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import os
WORKERS = int(os.environ.get("GROK_WORKERS", "28"))  # ceiling is the 8 GB RAM
GROK_SLOTS = threading.Semaphore(WORKERS)  # cap concurrent grok processes
STITCH_LOCK = threading.Lock()  # one big feather-stitch at a time (RAM)
MARGIN_FRAC = 0.04  # overlap margin per side, fraction of tile size

INSTRUCTIONS = {
    "canvas": (
        "make a mars terrain map, should be a cool topdown 2d view. generate it "
        "as one 16:9 canvas split 50/50 side by side: left half the terrain "
        "picture, right half its matching grayscale elevation heightmap, "
        "aligned. the heightmap must be matte pure elevation data (white = "
        "high, black = low) with no shading or lighting baked in. both halves "
        "must be rendered at the same resolution and the same level of fine "
        "detail - the heightmap as crisp as the terrain picture, its features "
        "matching precisely. "
        "print only the absolute file path of the saved image on the final "
        "line of your reply."
    ),
    "sky": (
        "generate a mars sky texture for a game sky dome, 16:9: hazy "
        "butterscotch-pink dusty martian sky, softly brighter near the sun, "
        "no ground, no horizon objects, just sky. print only the absolute "
        "file path of the saved image on the final line of your reply."
    ),
}

# Two-step canvas: the skill-loaded Grok agent authors the Imagine prompt
# (where its finetuned prompt-craft lives), then we fire the generation at the
# HTTP API ourselves with resolution enforced to 2k.
AUTHOR_INSTRUCTION = (
    "load your imagine and game-asset skills and use them to write one "
    "image-generation prompt, the way you would prompt your own image tool: "
    "a cool topdown 2d mars terrain map rendered as a single 16:9 canvas "
    "split 50/50 side by side, left half the terrain picture, right half its "
    "matching grayscale elevation heightmap (matte pure elevation data, "
    "white = high, black = low, no shading or lighting baked in), both halves "
    "aligned feature-for-feature at the same resolution and level of fine "
    "detail. make the terrain varied and interesting. do not call any tools "
    "and do not create any files. reply with exactly two lines:\n"
    "PROMPT: <the full image prompt>\n"
    "TITLE: <a short evocative name you invent for this terrain>"
)

API_URL = "https://api.x.ai/v1/images/generations"


def author_canvas_prompt() -> tuple:
    """Skill-loaded Grok agent writes the Imagine prompt + a world title."""
    with GROK_SLOTS:
        proc = subprocess.run(
            ["grok", "-p", AUTHOR_INSTRUCTION, "--always-approve", "--max-turns", "6"],
            capture_output=True, text=True, timeout=300,
        )
    out = (proc.stdout + "\n" + proc.stderr).strip()
    m = re.search(r"PROMPT:\s*(.+?)\s*TITLE:\s*(.+)", out, re.S)
    if proc.returncode != 0 or not m:
        raise RuntimeError(f"prompt authoring failed:\n{out[-1500:]}")
    return m.group(1).strip(), m.group(2).strip().splitlines()[0]


def api_generate(prompt: str, dst: Path, resolution: str = "2k",
                 aspect: str = "16:9") -> Path:
    """Direct xAI Imagine call with enforced resolution; downloads the image."""
    import json
    import urllib.request
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "model": "grok-imagine-image-quality",
            "prompt": prompt,
            "aspect_ratio": aspect,
            "resolution": resolution,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['XAI_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    url = data["data"][0]["url"]
    dst = dst.with_suffix(".png" if "png" in data["data"][0].get("mime_type", "png") else ".jpg")
    urllib.request.urlretrieve(url, dst)
    return dst


EDIT_INSTRUCTION = (
    "use your image edit tool on the image file at {path}. re-render this "
    "top-down mars terrain patch as if the same photo was taken at much "
    "higher resolution: keep every landform, crater, channel, shadow and "
    "color exactly in place with identical framing, only add fine realistic "
    "surface detail and sharpness. print only the absolute file path of the "
    "saved edited image on the final line of your reply."
)


def call_grok_full(instruction: str) -> tuple:
    """Run headless grok; return (last image path, full text output)."""
    with GROK_SLOTS:
        proc = subprocess.run(
            ["grok", "-p", instruction, "--always-approve", "--max-turns", "20"],
            capture_output=True, text=True, timeout=600,
        )
    out = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode != 0:
        raise RuntimeError(f"grok CLI failed:\n{out[-2000:]}")
    paths = re.findall(r"(/\S+\.(?:jpg|jpeg|png))", out)
    if not paths:
        raise RuntimeError(f"no image path in grok output:\n{out[-2000:]}")
    return Path(paths[-1]), out


def call_grok(instruction: str) -> Path:
    return call_grok_full(instruction)[0]


def split_canvas(src: Path, out_dir: Path) -> None:
    """Split a terrain|heightmap canvas into the two standard asset files.

    Saved verbatim — mesh-stability smoothing happens at build time
    (build_scene), never in the stored asset.
    """
    img = Image.open(src).convert("RGB")
    w, h = img.size
    img.crop((0, 0, w // 2, h)).save(out_dir / "terrain_texture.png")
    img.crop((w // 2, 0, w, h)).convert("L").save(out_dir / "heightmap.png")


def keep_original(src: Path, out_dir: Path, name: str) -> None:
    orig_dir = out_dir / "grok_originals"
    orig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, orig_dir / f"{name}{src.suffix.lower()}")


def canvas(out_dir: Path, levels: int) -> None:
    src = call_grok(INSTRUCTIONS["canvas"])
    keep_original(src, out_dir, "canvas")
    split_canvas(src, out_dir)
    print(f"canvas done: {out_dir}/terrain_texture.png + heightmap.png")


def sky(out_dir: Path, levels: int) -> None:
    src = call_grok(INSTRUCTIONS["sky"])
    keep_original(src, out_dir, "sky")
    Image.open(src).convert("RGB").save(out_dir / "MarsSky.png")
    print(f"sky done: {out_dir}/MarsSky.png")


# ---- recursive quadrant upscale -----------------------------------------

def quadrant_boxes(w: int, h: int):
    w2, h2 = w // 2, h // 2
    mx, my = int(w * MARGIN_FRAC), int(h * MARGIN_FRAC)
    return [
        (0, 0, w2 + mx, h2 + my),
        (w2 - mx, 0, w, h2 + my),
        (0, h2 - my, w2 + mx, h),
        (w2 - mx, h2 - my, w, h),
    ]


def color_match(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Align per-channel mean/std of arr to ref (keeps tone locked to parent)."""
    out = arr.astype(np.float32)
    ref = ref.astype(np.float32)
    for c in range(3):
        s = out[..., c].std() or 1.0
        out[..., c] = (out[..., c] - out[..., c].mean()) / s * ref[..., c].std() + ref[..., c].mean()
    return out.clip(0, 255)


def rerender_tile(crop: Image.Image, cache: Path) -> np.ndarray:
    """Grok image_edit re-render of one tile, resized to 2x, color-matched."""
    target = (crop.width * 2, crop.height * 2)
    if not cache.exists():
        src_path = cache.with_suffix(".src.png")
        crop.save(src_path)
        for attempt in range(3):
            try:
                result = call_grok(EDIT_INSTRUCTION.format(path=src_path))
                Image.open(result).convert("RGB").save(cache)
                break
            except Exception as e:
                if attempt < 2:
                    import time
                    time.sleep(15 * (attempt + 1))  # likely rate limit; back off
                else:  # degrade gracefully: plain upscale of the crop
                    print(f"WARN tile {cache.name} fell back to lanczos: {e}",
                          file=sys.stderr)
                    crop.resize(target, Image.LANCZOS).save(cache)
    out = np.asarray(Image.open(cache).convert("RGB").resize(target, Image.LANCZOS))
    ref = np.asarray(crop.resize(target, Image.LANCZOS))
    return color_match(out, ref)


def feather_stitch(children: list, boxes: list, parent_size: tuple, scale: int) -> np.ndarray:
    """Blend 4 fully-upscaled tiles into scale*parent_size with overlap ramps.

    Each child covers scale x its crop box; interior edges get a linear ramp
    across the (2 * margin * scale)-wide overlap band.
    """
    pw, ph = parent_size
    ow, oh = pw * scale, ph * scale
    rx = max(2, 2 * int(pw * MARGIN_FRAC) * scale)
    ry = max(2, 2 * int(ph * MARGIN_FRAC) * scale)
    STITCH_LOCK.acquire()
    acc = np.zeros((oh, ow, 3), dtype=np.float32)
    wsum = np.zeros((oh, ow, 1), dtype=np.float32)
    for arr, (x0, y0, x1, y1) in zip(children, boxes):
        th, tw = arr.shape[:2]
        wx = np.ones(tw, dtype=np.float32)
        wy = np.ones(th, dtype=np.float32)
        if x0 > 0:
            wx[:rx] = np.linspace(0, 1, rx)
        if x1 < pw:
            wx[-rx:] = np.minimum(wx[-rx:], np.linspace(1, 0, rx))
        if y0 > 0:
            wy[:ry] = np.linspace(0, 1, ry)
        if y1 < ph:
            wy[-ry:] = np.minimum(wy[-ry:], np.linspace(1, 0, ry))
        wmap = (wy[:, None] * wx[None, :])[..., None]
        ox, oy = x0 * scale, y0 * scale
        acc[oy:oy + th, ox:ox + tw] += arr * wmap
        wsum[oy:oy + th, ox:ox + tw] += wmap
    try:
        return acc / np.maximum(wsum, 1e-6)
    finally:
        STITCH_LOCK.release()


def upscale(out_dir: Path, levels: int) -> None:
    root = Image.open(out_dir / "terrain_texture.png").convert("RGB")
    work = out_dir / "upscale_work"
    work.mkdir(parents=True, exist_ok=True)

    def process(img: Image.Image, path_key: str, level: int) -> np.ndarray:
        if level == 0:
            return np.asarray(img, dtype=np.float32)
        boxes = quadrant_boxes(img.width, img.height)
        crops = [img.crop(b) for b in boxes]
        caches = [work / f"{path_key}q{q}.png" for q in range(4)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            tiles = list(ex.map(rerender_tile, crops, caches))
        with ThreadPoolExecutor(max_workers=4) as ex:
            deeper = list(ex.map(
                lambda qt: process(Image.fromarray(qt[1].astype(np.uint8)),
                                   f"{path_key}q{qt[0]}_", level - 1),
                enumerate(tiles)))
        return feather_stitch(deeper, boxes, (img.width, img.height), 2 ** level)

    result = process(root, "L", levels)
    out = out_dir / "terrain_hires.jpg"
    img = Image.fromarray(result.astype(np.uint8))
    if max(img.size) > 16384:  # GPU texture limit; oversized renders fail silently
        s = 16384 / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    img.save(out, quality=90)
    print(f"upscale done: {out} {img.width}x{img.height} "
          f"({2 ** levels}x linear from {root.width}x{root.height})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["canvas", "sky", "upscale"], required=True)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "assets")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"canvas": canvas, "sky": sky, "upscale": upscale}[args.mode](args.out, args.levels)


if __name__ == "__main__":
    main()
