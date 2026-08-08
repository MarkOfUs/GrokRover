# Endless Mars worlds from Grok Imagine

How this project turns single image-generation calls into rover-ready 3D
training terrain (`.usda` scenes for Isaac Sim).

## 1. The split-canvas trick: terrain + heightmap in ONE generation

A 3D terrain needs two things that must agree with each other: a color
texture (what the ground looks like) and a heightmap (its elevation). Generate
them with two separate image calls and they *drift* — the crater in the
picture won't match the pit in the elevation data.

The fix: **one 16:9 canvas, split 50/50 side by side** —
left half the terrain picture, right half its grayscale elevation heightmap
(white = high, black = low, matte, no baked lighting). Because both halves
come out of a single generation, the model keeps them aligned
feature-for-feature: every crater, canyon, and dune field appears in both
panels in the same place. We split the canvas down the middle and get a
perfectly matched texture + heightmap pair from **one API call**.

The heightmap drives a displaced grid mesh (Z-up, meters); the texture drapes
over it; deterministic Python assembles the USD scene with cameras and lights.
Grok also invents a title for each world — that's where the directory names
come from.

Prompt-craft notes that mattered:
- Ask for **matte, shading-free elevation** — otherwise the model bakes
  directional relief shading into the heightmap and the mesh inherits fake
  ridge noise.
- Demand **both halves at the same resolution and level of fine detail** —
  occasionally the heightmap half comes back soft and blobby. We also gate
  each generation on an edge-energy ratio between the two halves and retry
  soft ones automatically.

## 2. Recursive quadrant re-rendering: forcing Imagine past its resolution cap

Generation is two-step: a skill-loaded Grok agent authors the Imagine prompt
(its finetuned prompt-craft is the magic) plus a world title, then the prompt
is fired at the xAI image API directly with resolution enforced to 2k
(2816x1584). Even so, the terrain half is ~1400px across a 44 m world — crisp
from orbit, mush at rover-camera height. The image API tops out at 2k. But
there is `image_edit`.

So: split the terrain picture into **4 overlapping quadrants** and have Grok
re-render each one at full output resolution — "same photo, taken at higher
resolution: keep every landform exactly in place, only add fine detail." Each
tile comes back at ~2x the pixel density. Then **recurse**: each re-rendered
tile splits into 4 again. Three levels deep = 4 + 16 + 64 = 84 edit calls,
run ~24-28 at a time, and the 1408px panel becomes an **11264x12672** texture —
8x linear resolution, with genuinely new (model-imagined) surface detail at
every scale: cracked rock, gravel grain, dune ripples.

Keeping it seamless and faithful:
- **Overlapping crops** (4% margin) + **feather blending** across the overlap
  bands hide tile boundaries.
- **Per-tile color matching** (mean/std) back to the source crop stops tone
  drift as the recursion deepens.
- **The heightmap is never touched.** Only the color texture is re-rendered,
  so the 3D geometry — and the texture/elevation alignment — stays identical.

The upscaled micro-detail is invented, not recovered — macro features stay
put, but two runs grow different pebbles. For RL training that's free domain
randomization.

## Numbers per world

| Step | Calls | Output |
|---|---|---|
| Split canvas | 1 | aligned 960x1080 texture + heightmap |
| Recursive detail (3 levels) | 84 | 7680x8640 texture (~176 px/m) |
| Sky dome (shared, once ever) | 1 | MarsSky.png |

Each world lands as a self-contained `worlds/<slug>/` (scene.usda + assets,
original canvas kept in `assets/grok_originals/`) that drags into Isaac Sim
as a unit.
