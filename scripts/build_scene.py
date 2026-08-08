#!/usr/bin/env python3
"""Assemble mars_scene.usda from the Grok assets in assets/.

Inputs (from generate_assets.py): terrain_texture.png + heightmap.png (joint
canvas halves), terrain_hires.jpg (recursive quadrant upscale, optional),
MarsSky.png.

Small world = high texel density; the recursive Grok upscale multiplies the
ground texture resolution without ever touching the heightmap, so the mesh
stays identical. Textured sky dome + warm sun/fill; rover with solar panel
and mast cameras. Z-up, meters, Isaac-friendly. Gate with usdchecker +
usdrecord.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

WORLD_W = 44.0        # meters; height/depth follow the canvas panel aspect
HEIGHT_RANGE = 6.0    # meters black->white
GRID_STEP = 0.25      # target vertex spacing, meters
SPAWN_FLATTEN_R = 2.5 # landing-zone flatten radius, meters


def fmt(v):
    return f"({v[0]:.4g}, {v[1]:.4g}, {v[2]:.4g})"


def fmt2(v):
    return f"({v[0]:.4g}, {v[1]:.4g})"


def lookat_matrix(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.array(v, dtype=float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return [r, u, -f], eye


def matrix4_str(rows, eye):
    r, u, z = rows
    return (
        f"( ({r[0]:.4g}, {r[1]:.4g}, {r[2]:.4g}, 0), "
        f"({u[0]:.4g}, {u[1]:.4g}, {u[2]:.4g}, 0), "
        f"({z[0]:.4g}, {z[1]:.4g}, {z[2]:.4g}, 0), "
        f"({eye[0]:.4g}, {eye[1]:.4g}, {eye[2]:.4g}, 1) )"
    )


FACES = [
    (2, -1, [0, 3, 2, 1]), (2, +1, [4, 5, 6, 7]),
    (1, -1, [0, 1, 5, 4]), (1, +1, [2, 3, 7, 6]),
    (0, -1, [3, 0, 4, 7]), (0, +1, [1, 2, 6, 5]),
]


def box_mesh(name, mn, mx, mat, uvm=None, indent="    "):
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    bn, bst, bc, bi = [], [], [], []
    for axis, sign, vids in FACES:
        bc.append(4)
        bi.extend(vids)
        n = [0.0, 0.0, 0.0]
        n[axis] = float(sign)
        bn.extend([tuple(n)] * 4)
        ua, va = [(1, 2), (0, 2), (0, 1)][axis]
        for vid in vids:
            pt = p[vid]
            u, v = pt[ua] - mn[ua], pt[va] - mn[va]
            if uvm is None:
                u /= (mx[ua] - mn[ua]) or 1.0
                v /= (mx[va] - mn[va]) or 1.0
            else:
                u *= uvm
                v *= uvm
            bst.append((u, v))
    return f'''{indent}def Mesh "{name}" (
{indent}    prepend apiSchemas = ["MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    float3[] extent = [{fmt(mn)}, {fmt(mx)}]
{indent}    int[] faceVertexCounts = {bc}
{indent}    int[] faceVertexIndices = {bi}
{indent}    point3f[] points = [{", ".join(fmt(x) for x in p)}]
{indent}    normal3f[] normals = [{", ".join(fmt(n) for n in bn)}] (
{indent}        interpolation = "faceVarying"
{indent}    )
{indent}    texCoord2f[] primvars:st = [{", ".join(fmt2(x) for x in bst)}] (
{indent}        interpolation = "faceVarying"
{indent}    )
{indent}    uniform token subdivisionScheme = "none"
{indent}    uniform bool doubleSided = 1
{indent}    rel material:binding = </World/Materials/{mat}>
{indent}}}
'''


def tex_material(mid, tex, rough, metal=0.0):
    base = f"/World/Materials/{mid}"
    return f'''        def Material "{mid}"
        {{
            token outputs:surface.connect = <{base}/PBR.outputs:surface>

            def Shader "PBR"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{base}/Tex.outputs:rgb>
                float inputs:roughness = {rough}
                float inputs:metallic = {metal}
                token outputs:surface
            }}

            def Shader "stReader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}

            def Shader "Tex"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @assets/{tex}@ (
                    colorSpace = "sRGB"
                )
                float2 inputs:st.connect = <{base}/stReader.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float3 outputs:rgb
            }}
        }}
'''


def color_material(mid, rgb, rough, metal=0.0):
    return f'''        def Material "{mid}"
        {{
            token outputs:surface.connect = </World/Materials/{mid}/PBR.outputs:surface>

            def Shader "PBR"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {fmt(rgb)}
                float inputs:roughness = {rough}
                float inputs:metallic = {metal}
                token outputs:surface
            }}
        }}
'''


def build(assets: Path, out: Path) -> None:
    from PIL import ImageFilter
    macro = Image.open(assets / "terrain_texture.png").convert("RGB")
    hm_img = Image.open(assets / "heightmap.png").convert("L")
    # smooth only for meshing (jpg speckle -> bumpy quads); stored asset stays crisp
    hm_img = hm_img.filter(ImageFilter.MedianFilter(5))
    hm_img = Image.blend(hm_img.filter(ImageFilter.GaussianBlur(6)),
                         hm_img.filter(ImageFilter.GaussianBlur(1.5)), 0.35)
    pw, ph = macro.size
    W = WORLD_W
    D = WORLD_W * ph / pw

    # ground texture: recursive Grok upscale if available, else the raw panel
    ground_tex = "terrain_hires.jpg" if (assets / "terrain_hires.jpg").exists() else "terrain_texture.png"
    if ground_tex == "terrain_texture.png":
        print("NOTE: terrain_hires.jpg missing, binding base-res terrain_texture.png")

    # --- heightfield ------------------------------------------------------
    hm = np.asarray(hm_img, dtype=np.float32) / 255.0
    nx = int(W / GRID_STEP) + 1
    ny = int(D / GRID_STEP) + 1
    xs = np.linspace(0, W, nx)
    ys = np.linspace(0, D, ny)
    ui = np.clip((xs / W) * (hm.shape[1] - 1), 0, hm.shape[1] - 1).astype(int)
    vi = np.clip((1 - ys / D) * (hm.shape[0] - 1), 0, hm.shape[0] - 1).astype(int)
    H = hm[np.ix_(vi, ui)] * HEIGHT_RANGE

    def height_at(x, y):
        i = np.clip(x / W * (nx - 1), 0, nx - 1)
        j = np.clip(y / D * (ny - 1), 0, ny - 1)
        i0, j0 = int(i), int(j)
        i1, j1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1)
        fi, fj = i - i0, j - j0
        return (H[j0, i0] * (1 - fi) * (1 - fj) + H[j0, i1] * fi * (1 - fj)
                + H[j1, i0] * (1 - fi) * fj + H[j1, i1] * fi * fj)

    # rover spawn: flattest interior spot
    best, spawn = None, (W / 2, D / 2)
    win = 4
    for j in range(int(ny * 0.2), int(ny * 0.8), 2):
        for i in range(int(nx * 0.2), int(nx * 0.8), 2):
            v = H[j - win:j + win + 1, i - win:i + win + 1].std()
            if best is None or v < best:
                best, spawn = v, (xs[i], ys[j])
    rx, ry = spawn
    rz = height_at(rx, ry)

    # flatten landing zone around the spawn (v1 trick, keeps the drop stable)
    gx, gy = np.meshgrid(xs, ys)
    blend = np.exp(-(((gx - rx) ** 2 + (gy - ry) ** 2) / SPAWN_FLATTEN_R ** 2))
    H = H * (1 - blend) + rz * blend

    yaw = math.atan2(D / 2 - ry, W / 2 - rx)
    yaw_deg = math.degrees(yaw)

    pts, st = [], []
    for j in range(ny):
        for i in range(nx):
            pts.append(f"({xs[i]:.4g}, {ys[j]:.4g}, {H[j, i]:.4g})")
            st.append(f"({xs[i] / W:.4g}, {ys[j] / D:.4g})")
    counts, indices = [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            counts.append("4")
            indices += [str(a), str(a + 1), str(a + nx + 1), str(a + nx)]
    dzdx = np.gradient(H, xs, axis=1)
    dzdy = np.gradient(H, ys, axis=0)
    nrm = []
    for j in range(ny):
        for i in range(nx):
            n = np.array([-dzdx[j, i], -dzdy[j, i], 1.0])
            n /= np.linalg.norm(n)
            nrm.append(f"({n[0]:.4g}, {n[1]:.4g}, {n[2]:.4g})")

    # --- cameras ----------------------------------------------------------
    overhead = matrix4_str(*lookat_matrix((W / 2, D / 2, 42), (W / 2, D / 2 + 0.01, 0)))
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0])
    left = np.array([-fwd[1], fwd[0], 0])
    hero_eye = np.array([rx, ry, rz]) - 6.5 * fwd + 3.5 * left + np.array([0, 0, 2.4])
    hero = matrix4_str(*lookat_matrix(hero_eye, (rx, ry, rz + 0.9)))

    parts = [f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Mars rover scene - Grok Imagine terrain/heightmap/tiles, deterministic assembly"
)

def Xform "World" (
    kind = "assembly"
)
{{
    def Mesh "Terrain" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        float3[] extent = [(0, 0, {float(H.min()):.4g}), ({W:.4g}, {D:.4g}, {float(H.max()):.4g})]
        int[] faceVertexCounts = [{", ".join(counts)}]
        int[] faceVertexIndices = [{", ".join(indices)}]
        point3f[] points = [{", ".join(pts)}]
        normal3f[] normals = [{", ".join(nrm)}] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [{", ".join(st)}] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        uniform bool doubleSided = 1
        rel material:binding = </World/Materials/MarsGround>
    }}
''']

    parts.append(f'''    def Scope "Cameras"
    {{
        def Camera "OverheadCam"
        {{
            float focalLength = 16
            float2 clippingRange = (0.5, 300)
            matrix4d xformOp:transform = {overhead}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}

        def Camera "HeroCam"
        {{
            float focalLength = 21
            float2 clippingRange = (0.05, 300)
            matrix4d xformOp:transform = {hero}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}
    }}

    def DomeLight "SkyLight"
    {{
        float inputs:intensity = 1.1
        color3f inputs:color = (1, 0.88, 0.75)
        asset inputs:texture:file = @assets/MarsSky.png@
    }}

    def DistantLight "Sun"
    {{
        float inputs:intensity = 3.2
        color3f inputs:color = (1, 0.82, 0.62)
        float3 xformOp:rotateXYZ = (-62, 0, 35)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}

    def DistantLight "Fill"
    {{
        float inputs:intensity = 0.9
        color3f inputs:color = (1, 0.75, 0.6)
        float3 xformOp:rotateXYZ = (-75, 0, -150)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}

    def Scope "Materials"
    {{
''')
    parts.append(tex_material("MarsGround", ground_tex, 0.9))
    parts.append('''    }
}
''')

    out.write_text("".join(parts))
    print(f"wrote {out} | world {W:.0f}x{D:.0f} m | {nx * ny} verts | "
          f"spawn (flattened) at ({rx:.1f}, {ry:.1f}, {rz:.2f})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=Path(__file__).parent.parent / "assets")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "mars_scene.usda")
    args = ap.parse_args()
    build(args.assets, args.out)
