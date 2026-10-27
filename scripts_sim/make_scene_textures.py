"""Procedural textures for the bookshelf scene (assembly/scene_bookshelf.xml): numpy + Pillow, deterministic.

    python scripts_sim/make_scene_textures.py             # -> assembly/textures/*.png
    python scripts_sim/make_scene_textures.py --out DIR   # write somewhere else

    desk_wood.png     1024^2  oak planks, grain along u (= world x on the desk top), 5 planks across v
    shelf_wood.png     512^2  walnut board, grain along u        (shelf boards: grain along x)
    shelf_wood_v.png   512^2  the same turned 90 degrees         (side and back panels: grain along z)
    floor.png         1024^2  light grey concrete (grey levels, tinted by the material rgba)
    wall.png           512^2  matte plaster (grey levels, tinted by the material rgba)
    book_cloth.png     256^2  fine cloth / leather grain (grey levels, tinted per book by geom rgba)
    book_paper.png     256^2  page block seen from above: page edges as fine lines along v (grey levels)

All noise is white noise low-passed in Fourier space, so each texture tiles seamlessly, and every random
draw comes from a fixed seed: re-running reproduces the same files. The whole folder stays under 2.5 MB.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assembly/textures"


# ---------------------------------------------------------------------------------------------------------
# noise helpers (all periodic in both axes)
# ---------------------------------------------------------------------------------------------------------
def noise(rng, h, w, sx, sy):
    """Zero-mean, unit-variance noise with Gaussian correlation lengths sx (along u, columns) and sy (along v,
    rows) in pixels. A length of 0 leaves that axis white."""
    spec = np.fft.rfft2(rng.standard_normal((h, w)))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    spec *= np.exp(-2.0 * np.pi ** 2 * ((sx * fx) ** 2 + (sy * fy) ** 2))
    out = np.fft.irfft2(spec, s=(h, w))
    return (out - out.mean()) / out.std()


def fbm(rng, h, w, sx, sy, octaves=4, gain=0.55):
    """Sum of noise octaves, the correlation length halving each time; unit variance."""
    out, amp, norm = 0.0, 1.0, 0.0
    for k in range(octaves):
        out = out + amp * noise(rng, h, w, sx / 2 ** k, sy / 2 ** k)
        norm += amp ** 2
        amp *= gain
    return out / np.sqrt(norm)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def to_png(path, img):
    """img in 0..1, (h, w) grey levels or (h, w, 3) RGB -> 8-bit PNG."""
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(path, optimize=True, compress_level=9)
    print(f"  {path.name:18s} {im.width}x{im.height} {im.mode:3s} {path.stat().st_size / 1024:6.0f} kB")


# ---------------------------------------------------------------------------------------------------------
# wood
# ---------------------------------------------------------------------------------------------------------
def wood(rng, size, planks, light, dark, rings_per_plank, seam=0.35, tint_spread=0.07, hue_spread=0.03):
    """Planks side by side (rows), grain along the columns. Growth rings are the contour lines of a warped
    coordinate, so they wander and arch like flat-sawn boards; late wood is the dark edge of each ring; fine
    fibres and sparse pores run along the grain. Returns (size, size, 3) in 0..1."""
    h = w = size
    v = (np.arange(h)[:, None] + 0.5) / h
    u = (np.arange(w)[None, :] + 0.5) / w
    idx = np.minimum((v * planks).astype(int), planks - 1)  # plank of each row
    vp = v * planks - idx  # 0..1 across the plank

    def per_plank(lo, hi):
        return rng.uniform(lo, hi, planks)[idx]

    # ring coordinate: across-plank position + slow wander + one arch (cathedral) per plank, periodic along u;
    # the ring spacing itself drifts across the plank so the rings are not evenly spaced
    wander = fbm(rng, h, w, sx=w / 2.0, sy=h / planks / 3.0, octaves=3)
    spacing = noise(rng, h, w, sx=w / 1.5, sy=h / planks / 2.5)
    u0 = per_plank(0.0, 1.0)
    arch = 0.5 * (1.0 - np.cos(2.0 * np.pi * (u - u0)))
    ring = (vp * per_plank(0.85, 1.15) * rings_per_plank + per_plank(0, 1)
            + 0.75 * wander * per_plank(0.6, 1.4) + 0.35 * spacing + 2.2 * arch * per_plank(-1.0, 1.0))
    f = ring - np.floor(ring)
    late = f ** 2.2  # slow darkening, then an abrupt new ring
    fibre = noise(rng, h, w, sx=w / 6.0, sy=0.6)  # long fine streaks
    pores = smoothstep(1.7, 3.0, noise(rng, h, w, sx=w / 60.0, sy=0.5))  # short dark dashes
    blotch = fbm(rng, h, w, sx=w / 3.0, sy=h / 3.0, octaves=3)  # slow brightness drift

    t = 0.20 + 0.36 * late + 0.13 * fibre + 0.05 * blotch + 0.32 * pores
    t = np.clip(t, 0.0, 1.0)[..., None]
    light, dark = np.asarray(light), np.asarray(dark)
    rgb = light * (1.0 - t) + dark * t
    warm = per_plank(-hue_spread, hue_spread)  # each plank a little redder or yellower than its neighbours
    rgb = rgb * per_plank(1.0 - tint_spread, 1.0 + tint_spread)[..., None]
    rgb = rgb * np.stack([1.0 + warm, np.ones_like(warm), 1.0 - warm], -1)
    if planks > 1:  # glue lines between the planks
        edge = np.minimum(vp, 1.0 - vp) * h / planks  # pixels to the nearest seam
        rgb = rgb * (1.0 - seam * np.exp(-(edge / 1.2) ** 2))[..., None]
    return np.clip(rgb, 0.0, 1.0)


def desk_wood():
    rng = np.random.default_rng(11)
    return wood(rng, 1024, planks=5, light=(0.76, 0.62, 0.45), dark=(0.55, 0.40, 0.26), rings_per_plank=14.0)


def shelf_wood():
    rng = np.random.default_rng(23)
    return wood(rng, 512, planks=1, light=(0.52, 0.34, 0.20), dark=(0.30, 0.18, 0.10), rings_per_plank=20.0,
                tint_spread=0.04)


# ---------------------------------------------------------------------------------------------------------
# room and book surfaces (grey levels; colour comes from the material / geom rgba)
# ---------------------------------------------------------------------------------------------------------
def concrete(size=1024):
    rng = np.random.default_rng(31)
    h = w = size
    g = (0.80 + 0.045 * fbm(rng, h, w, size / 4.0, size / 4.0, octaves=5)  # cloudy patches
         + 0.018 * noise(rng, h, w, 6.0, 6.0)  # trowel marks
         + 0.030 * noise(rng, h, w, 0.8, 0.8))  # sand grain
    pits = smoothstep(2.6, 3.4, noise(rng, h, w, 1.4, 1.4))  # tiny air holes
    return g - 0.16 * pits


def plaster(size=512):
    rng = np.random.default_rng(41)
    h = w = size
    return (0.93 + 0.012 * fbm(rng, h, w, size / 5.0, size / 5.0, octaves=4)
            + 0.014 * noise(rng, h, w, 1.2, 1.2) + 0.006 * noise(rng, h, w, 0.0, 0.0))


def cloth(size=256):
    """Book cloth / leather: slightly directional grain plus a faint weave."""
    rng = np.random.default_rng(53)
    h = w = size
    y, x = np.mgrid[0:h, 0:w]
    weave = 0.5 * (np.cos(2 * np.pi * x / 4.0) * np.cos(2 * np.pi * y / 4.0))
    return (0.84 + 0.045 * noise(rng, h, w, 0.8, 0.8) + 0.04 * noise(rng, h, w, 3.0, 1.2)
            + 0.025 * weave + 0.02 * fbm(rng, h, w, size / 6.0, size / 6.0, octaves=3))


def paper(size=256):
    """Top of a page block: pages are lines along v, with a little unevenness between them."""
    rng = np.random.default_rng(61)
    h = w = size
    x = np.arange(w)[None, :]
    lines = 0.5 * np.cos(2 * np.pi * x / 5.0 + 2.0 * noise(rng, 1, w, 1.5, 0.0))  # period 5 px, wobbling
    return 0.90 + 0.035 * lines + 0.02 * noise(rng, h, w, 0.8, 6.0) + 0.012 * noise(rng, h, w, 0.8, 0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default: assembly/textures)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"writing {args.out}")
    shelf = shelf_wood()
    to_png(args.out / "desk_wood.png", desk_wood())
    to_png(args.out / "shelf_wood.png", shelf)
    to_png(args.out / "shelf_wood_v.png", np.rot90(shelf))
    to_png(args.out / "floor.png", concrete())
    to_png(args.out / "wall.png", plaster())
    to_png(args.out / "book_cloth.png", cloth())
    to_png(args.out / "book_paper.png", paper())
    total = sum(p.stat().st_size for p in args.out.glob("*.png"))
    print(f"total {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
