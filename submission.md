# GrokRover: Hackathon Submission

## Tagline

Real Mars terrain, imagined past the resolution limit by Grok, and an RL rover that learns to drive it.

## Project Description

GrokRover turns real orbital data of Mars into drivable simulation worlds, then trains a rover to drive them with reinforcement learning.

The core problem: the best data that exists for Mars stops short of what a rover needs. HiRISE imagery of Jezero Crater resolves 25 cm, but elevation is estimated by comparing photos of the same ground from different angles, giving one height value per meter. Rover-scale rocks are visible in the photos but flat in the 3D terrain, so simulation built from raw data is smooth ramps with nothing to learn on.

Our pipeline closes that gap. The source data is the real thing: imagery captured by the HiRISE camera aboard NASA's Mars Reconnaissance Orbiter, assembled by USGS Astrogeology into the hazard basemap for the Mars 2020 mission. These are the exact maps the Perseverance rover matched its descent cameras against to land safely in February 2021. We crop a patch of that basemap and compose a split canvas: real photo on the left, real elevation on the right. A skill-loaded Grok agent authors its own edit prompt, and Grok Imagine redraws both halves jointly, adding rover-scale rocks, ripples and regolith to the photo AND the heightmap, effectively doing zero-shot single-image elevation estimation. Automated gates keep the imagination honest, rejecting edits whose heightmap stays flat or drifts from the real elevation. Then a recursive re-render pushes the texture to rover-camera density: the image is split into four overlapping quadrants, each re-rendered with Grok Imagine image editing at double detail, and the process recurses three levels deep, 84 image-edit calls per world, before the tiles are color-matched and feather-stitched back together for 8x resolution. A deterministic builder turns the two images into USD scenes. The library is 30+ worlds, each with full real-data provenance.

On these worlds we train GrokRover, a rover, in NVIDIA Isaac Sim with massively parallel reinforcement learning: hundreds of environments learning goal-directed driving on terrain where naive driving fails.

Key learning: image models can be a terrain engine. One model gave us unlimited, diverse, real-place training worlds that would each take an artist days to hand-build, and terrain diversity is exactly what RL needs to learn driving that transfers to ground it has never seen.
