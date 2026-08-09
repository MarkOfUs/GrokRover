# MarsProject v2 — real-data worlds (planned pipeline)

Same pipeline as [README](README.md), but instead of Grok imagining a world
from scratch, we seed it with **real Mars data** and let Grok fill in the
detail no instrument can measure. Input: a coordinate on Mars. Output: a
drivable, metrically grounded USD world of that actual place.

![input to world](docs/img/placeholder_pipeline_story.png)
*(placeholder — real seed → Grok canvas → drivable world)*

## The data, and why it's the best there is

Two co-registered public products per site (our example: the USGS Mars 2020
mosaics of Jezero Crater — the exact maps NASA used to land Perseverance):

- **Photo — HiRISE orthoimage, 25 cm/px.** HiRISE (on Mars Reconnaissance
  Orbiter, ~300 km up) is the sharpest camera ever flown to another planet.
- **Elevation — stereo DTM, ~1 m/px.** HiRISE photographs the same ground
  twice from different orbital angles; the per-pixel parallax between the two
  images is converted to height (stereo photogrammetry), then anchored to the
  MOLA laser-altimeter global datum. Vertical precision: tens of cm.

This is the ceiling, not a budget cut: stereo cannot resolve elevation much
below the source image resolution, so **sub-meter Mars terrain does not exist
and cannot be collected from orbit** (rover navcams see cm-scale, but only
along their own tracks). A 30 cm rock — the thing that breaks rovers — is
visible in the photo but absent from the elevation. That gap is what we fill.

For scale: MOLA, the usual "Mars topography" answer, is 463 m/px — our whole
44 m world fits inside one of its pixels.

## Pipeline

```
coordinates (e.g. Jezero delta)
   │  ranged HTTP reads (rasterio /vsicurl/) — never download the multi-GB mosaics
   ▼
seed canvas: [ tinted ortho photo | DTM as grayscale heightmap ]   ← scripts/seed_canvas.py
   │  one grok image_edit call, same prompt-craft as the v1 canvas
   ▼
enhanced canvas: same 50/50 split, same landforms, + rover-scale rocks,
ripples and regolith drawn into BOTH halves
   │  split (unchanged v1 code)
   ▼
terrain_texture.png + heightmap.png → recursive quadrant upscale (texture only,
unchanged) → build_scene.py → worlds/<slug>/scene.usda + proof renders
```

Steps after the seed are the v1 pipeline verbatim — real-data mode is one new
input stage, nothing downstream changes.

![seed canvas](docs/img/placeholder_seed_canvas.png)
*(placeholder — what Grok receives: sharp real photo, blurry real elevation)*

## How the fine resolution gets added

Grok performs **photo-guided elevation super-resolution**: the photo says
where rocks, dune ripples and rubble are; Grok redraws the heightmap with
that sub-meter geometry added, on top of the true 1 m topography. The added
detail is *plausible*, not measured — but the large-scale structure stays
real, and the photo keeps the invention honest.

Prompt-craft matters (found the hard way): asking Grok to "preserve every
shadow" yields a desaturated photo, not elevation (r = +0.96 vs photo
luminance, garbage as height). Asking in the v1 canvas language — *matte pure
elevation, white = high, no shading* — yields a real heightmap: **r = +0.82
against the true DTM** while gaining crisp rock detail.

Quality gates per world: the v1 edge-energy detail ratio, plus a correlation
check (height panel must track the real DTM, not the photo).

![strategies](docs/img/placeholder_heightmap_strategies.png)
*(placeholder — raw data only vs Grok-drawn vs enhanced: smooth ramps vs fake
shadow-mountains vs real structure + hazards)*

## Scale

A 160×180 m real patch with ~27 m relief maps to the 44 m world with 6 m
height range — a near-faithful scale model (slopes come out ~20% gentler than
reality). Planned: pass the patch's true dimensions from the DTM into
build_scene for metrically exact worlds — rover wheelbase vs rock size is the
interaction RL actually cares about.

## Why it matters

Public data tells you where the mountains are. Nothing tells you where the
rocks are — and rocks are what break rovers. We turn NASA's best orbital maps
into drivable, rover-scale worlds of real Mars locations: train on synthetic
variants, evaluate on true Jezero topography.
