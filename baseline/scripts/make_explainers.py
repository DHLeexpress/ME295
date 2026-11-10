"""Figures and clips that explain the headline results table (what each column tests, why each row scores what it does).

    python baseline/scripts/make_explainers.py suites        # -> results/fig_suites.{png,pdf}: one test case per column
    python baseline/scripts/make_explainers.py modes         # -> results/fig_outcomes.{png,pdf} + media/outcomes.{mp4,gif}
    python baseline/scripts/make_explainers.py reel-assets   # -> reel/results_60s/assets/*.jpg + src/clips.js (sprites)
    python baseline/scripts/make_explainers.py reel-timing   # after `reel voice`: clamp word times to their line
    python baseline/scripts/make_explainers.py reel-publish  # after `reel export`: results/media/reel_results_60s.mp4
                                                             # (720p) + reel_results_60s_strip.jpg

`modes` and `reel-assets` use the high-resolution media re-runs in baseline/data/media_runs/ (git-ignored), made with
    python baseline/scripts/evaluate.py --policy <p> --suites <s> --indices <i,...> --media --out baseline/data/media_runs
They replay seeds of the evaluation suites; the policy samples its actions, so a re-run is a new episode of the same
test case. They illustrate the failure modes and are never counted in the results (results/eval/ holds those).
Re-runs used (each with --media --videos 0 --out baseline/data/media_runs; servers as in README.md):
    pi05_ft     nominal 0,1   random 5,10,27,29,41,47   narrow 1,2,21,23   heavy_lowmu 0,3,4,7
    pi05_droid  nominal 0,1,2   random 1
    expert      nominal 0   random 5,27,47   narrow 1,23   heavy_lowmu 0,3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "baseline" / "results"
RUNS = ROOT / "baseline" / "data" / "media_runs"
REEL = ROOT / "baseline" / "reel" / "results_60s"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline" / "scripts"))

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"pi05_ft": "#2a78d6", "pi05_droid": "#eb6834", "expert": "#1baf7a"}   # same slots as make_figures.py
NAME = {"pi05_ft": "π0.5 fine-tuned", "pi05_droid": "π0.5-DROID zero-shot", "expert": "scripted expert"}
BLADE_MM = 3.5      # blade + skin through the pad (embodiment/check.py)

# one representative test case per column of the results table (seed index in the evaluation suite)
SUITE_CASES = [
    ("nominal", 0, "Nominal (simulator scene)", "dark-blue 40 mm book, 10 mm gaps, 0.8 kg;\nonly the start pose changes"),
    ("random", 5, "Randomized layouts", "any 35-45 mm book, 8-12 mm gaps, 0.4-1.2 kg,\n8 colours, sometimes a 2nd gapped book"),
    ("narrow", 2, "Narrow gaps", "6.5-7.5 mm gaps, narrower than\nany demonstration"),
    ("heavy_lowmu", 0, "Heavy + slippery", "1.6-2.4 kg book, skin friction 0.5\ninstead of 1.0"),
]


# ------------------------------------------------------------------------------------------------ rendering helpers
def _gl():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("EGL_GPU", "2"))


def free_cam_pose(center, view):
    """Position and (right, up, forward) axes of a MuJoCo free camera (mjv_updateCamera conventions)."""
    az, el = np.deg2rad(view["azimuth"]), np.deg2rad(view["elevation"])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    lookat = np.asarray(center, float) + np.asarray(view["offset"], float)
    return lookat - view["distance"] * fwd, np.cross(fwd, up), up, fwd


def project(p, pose, fovy_deg, w, h):
    pos, right, up, fwd = pose
    d = np.asarray(p, float) - pos
    f = 0.5 * h / np.tan(np.deg2rad(fovy_deg) / 2)
    z = d @ fwd
    return np.array([w / 2 + f * (d @ right) / z, h / 2 - f * (d @ up) / z])


def load_jsonl(path: Path):
    return {r["index"]: r for r in (json.loads(x) for x in open(path))} if path.exists() else {}


def table_cell(policy: str, suite: str) -> str:
    rows = load_jsonl(RES / "eval" / {"pi05_ft": "pi05_ft_step9999", "expert": "expert",
                                      "pi05_droid": "pi05_droid"}[policy] / f"{suite}.jsonl")
    rows = [r for r in rows.values() if "error" not in r]
    return f"{sum(r['success'] for r in rows)}/{len(rows)}" if rows else "–"


def measured_gaps(env) -> dict:
    """Air between the designated book and its neighbours in the row (mm), from the simulated geometry."""
    m, d, des = env.m, env.rob.d, env.info.designated
    gd = m.geom(des).id
    cx, hx = d.geom_xpos[gd][0], m.geom_size[gd][0]
    left, right = [], []
    for g in range(m.ngeom):
        n = m.geom(g).name
        if not n.startswith("book") or g == gd or abs(d.geom_xpos[g][1] - d.geom_xpos[gd][1]) > 0.05 \
                or abs(d.geom_xpos[g][2] - d.geom_xpos[gd][2]) > 0.1:
            continue
        x, h = d.geom_xpos[g][0], m.geom_size[g][0]
        (left if x < cx else right).append(abs(x - cx) - h - hx)
    return {"gap_left_mm": 1e3 * min(left), "gap_right_mm": 1e3 * min(right)}


# ------------------------------------------------------------------------------------------------ fig_suites
def fig_suites():
    _gl()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    from evaluate import make_case
    from baseline.sim.env import EnvConfig, LibrarianPickEnv
    from baseline.sim.media import VIEWS, free_camera

    W_, H_ = 1200, 900
    view = dict(VIEWS["top"], offset=(0.0, -0.01, 0.10), distance=0.40)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.4), facecolor=SURFACE)
    for ax, (suite, idx, title, sub) in zip(axes, SUITE_CASES):
        v, kw = make_case(suite, idx)
        env = LibrarianPickEnv(v, EnvConfig(**kw))
        m = env.m
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W_), max(m.vis.global_.offheight, H_)
        r = mujoco.Renderer(m, H_, W_)
        c = env.info.center
        r.update_scene(env.rob.d, camera=free_camera(c, view))
        img = r.render().copy()
        r.close()
        met = env.metrics()
        met.update(measured_gaps(env))
        env.close()
        pose = free_cam_pose(c, view)
        ax.imshow(img)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # designated book: a label above its spine; gaps: the measured air on each side
        half = env.info.half
        front_z = c[2] + half[2]
        top = project([c[0], c[1] - half[1], front_z], pose, m.vis.global_.fovy, W_, H_)
        ax.annotate(f"pick up the {v.color} book\n{met['thickness_mm']:.0f} mm, {met['mass_kg']:.1f} kg",
                    xy=top, xytext=(top[0], 95), ha="center", va="top", fontsize=10.5, color=TEXT, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.6))
        for side, key in ((-1, "gap_left_mm"), (1, "gap_right_mm")):
            g = met[key] / 1e3
            xg = c[0] + side * (half[0] + g / 2)
            p = project([xg, c[1] - half[1], front_z], pose, m.vis.global_.fovy, W_, H_)
            ax.annotate(f"{met[key]:.1f} mm", xy=p, xytext=(p[0] + side * 150, H_ - 90), ha="center", fontsize=10.5,
                        color=TEXT, bbox=dict(boxstyle="round,pad=0.25", fc="#fff4c2", ec="none"),
                        arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.4))
        ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT, loc="left", pad=26)
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=9.5, color=TEXT2, va="bottom")
        # the table's row values for this column
        lines = [(f"air left beside the {BLADE_MM} mm blades: {met['gap_left_mm'] - BLADE_MM:.1f} mm | "
                  f"{met['gap_right_mm'] - BLADE_MM:.1f} mm", TEXT2)]
        lines += [(f"{NAME[p]}: {table_cell(p, suite)}", COLOR[p]) for p in ("pi05_ft", "expert")]
        if suite in ("nominal", "random"):
            lines.append((f"{NAME['pi05_droid']}: {table_cell('pi05_droid', suite)}", COLOR["pi05_droid"]))
        for k, (s, col) in enumerate(lines):
            ax.text(0.0, -0.035 - 0.07 * k, ("■ " if k else "") + s, transform=ax.transAxes, fontsize=10.5,
                    color=TEXT if k else TEXT2, va="top")
            if k:   # the colour square carries identity; the text stays in ink
                ax.text(0.0, -0.035 - 0.07 * k, "■", transform=ax.transAxes, fontsize=10.5, color=col, va="top")
    fig.suptitle("What each column of the results table tests (one test case per suite, seen from above the shelf; "
                 "the gripper enters from the bottom)", fontsize=12, color=TEXT2, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.835, bottom=0.2, wspace=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"fig_suites.{ext}", dpi=110, facecolor=SURFACE)
    print("wrote", RES / "fig_suites.png")



# ------------------------------------------------------------------------------------------------ reel assets
# clip key -> (policy, episode, view, t_start, t_end, frames): shown on the reel's monitors (time-lapsed to fit a scene)
REEL_CLIPS = {
    "hook": ("pi05_ft", "nominal_001", "close", 0.0, 8.0, 32),
    "cell": ("pi05_ft", "nominal_001", "wide", 2.0, 14.5, 32),
    "zero": ("pi05_droid", "nominal_000", "wide", 8.5, 15.5, 32),
    "tuned": ("pi05_ft", "nominal_001", "close", 6.5, 14.5, 32),
    "push": ("pi05_ft", "random_005", "top", 3.5, 8.0, 24),
    "shove": ("pi05_ft", "narrow_002", "top", 4.0, 10.0, 20),
    "slip": ("pi05_ft", "heavy_lowmu_000", "close", 8.0, 21.0, 32),
    "slipx": ("expert", "heavy_lowmu_000", "close", 5.0, 16.4, 24),
}
# one thumbnail per row of the results table (the test case of fig_suites, before the robot moves)
REEL_STILLS = {"thumb_nominal": ("pi05_ft", "nominal_001", "top", 0.5), "thumb_random": ("pi05_ft", "random_005", "top", 0.5),
               "thumb_narrow": ("pi05_ft", "narrow_002", "top", 0.5), "thumb_heavy": ("pi05_ft", "heavy_lowmu_000", "top", 0.5)}
SPRITE_W, SPRITE_H, SPRITE_COLS = 560, 420, 8


def media_fps(policy: str) -> float:
    return 15.0 if policy == "pi05_droid" else 10.0


def read_frames(policy: str, episode: str, view: str):
    import imageio.v2 as iio
    return iio.mimread(RUNS / policy / "media" / f"{episode}_{view}.mp4", memtest=False)


def reel_assets():
    from PIL import Image
    out = REEL / "assets"
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for key, (pol, ep, view, t0, t1, n) in REEL_CLIPS.items():
        fr = read_frames(pol, ep, view)
        hz = media_fps(pol)
        idx = [min(len(fr) - 1, int(round((t0 + (t1 - t0) * k / (n - 1)) * hz))) for k in range(n)]
        rows = -(-n // SPRITE_COLS)
        sheet = Image.new("RGB", (SPRITE_W * SPRITE_COLS, SPRITE_H * rows))
        for k, i in enumerate(idx):
            sheet.paste(Image.fromarray(fr[i]).resize((SPRITE_W, SPRITE_H), Image.LANCZOS),
                        ((k % SPRITE_COLS) * SPRITE_W, (k // SPRITE_COLS) * SPRITE_H))
        sheet.save(out / f"{key}.jpg", quality=80, optimize=True)
        meta[key] = {"n": n, "cols": SPRITE_COLS, "fw": SPRITE_W, "fh": SPRITE_H, "src": f"{pol}/{ep}_{view}",
                     "t0": t0, "t1": t1}
    for key, (pol, ep, view, t) in REEL_STILLS.items():
        fr = read_frames(pol, ep, view)
        Image.fromarray(fr[min(len(fr) - 1, int(t * media_fps(pol)))]).resize((SPRITE_W, SPRITE_H), Image.LANCZOS) \
            .save(out / f"{key}.jpg", quality=85, optimize=True)
    js = ("// generated by baseline/scripts/make_explainers.py reel-assets: sprite sheets of the media re-runs\n"
          f"const RL_CLIPS = {json.dumps(meta)};\n")
    (REEL / "src" / "clips.js").write_text(js)
    total = sum(f.stat().st_size for f in out.glob("*.jpg")) / 1e6
    print(f"wrote {len(meta)} sprite sheets + {len(REEL_STILLS)} stills to {out} ({total:.1f} MB)")



# ------------------------------------------------------------------------------------------------ fig_outcomes + clip
# one row per cell of the table that needs explaining: (policy, episode of the media re-runs, view, key frame times,
# title, what happens, the table cell it explains). Mechanism numbers below are from these re-runs (see main()).
OUTCOME_ROWS = [
    ("pi05_droid", "nominal_000", "wide", [1.0, 7.0, 12.0, 15.4], (0.0, 15.5),
     "π0.5-DROID zero-shot · nominal",
     "wanders above the shelf with the jaw open, then sweeps into it:\nwrist force > 150 N = UR protective stop",
     "0/20 nominal, 0/20 randomized (19 of 20 stop like this)"),
    ("pi05_ft", "nominal_001", "close", [4.0, 7.6, 11.9, 14.4], (3.0, 14.5),
     "π0.5 fine-tuned · nominal (its training scene)",
     "blades into the 10 mm gaps, squeeze, pull out, lift, hold 1 s",
     "50/50, neighbours untouched (median 0.0 mm)"),
    ("pi05_ft", "random_005", "top", [4.0, 5.5, 7.0, 20.0], (3.0, 12.0),
     "π0.5 fine-tuned · randomized, 45 mm book",
     "opens only 2 mm wider than the book (expert: 6 mm); the blade tips\ncatch the spine and push it 56 mm deeper; the pads never touch it",
     "36/50: 7 of the 14 failures are push-ins like this"),
    ("pi05_ft", "narrow_002", "top", [4.0, 6.0, 8.0, 12.0], (3.0, 13.0),
     "π0.5 fine-tuned · narrow gaps (7 mm)",
     "opens to its 40 mm-book setting around a 36 mm book: the blades need\n53 mm, the slot is 51 mm, so a neighbour is shoved 56 mm (still a success)",
     "22/30, but a neighbour moved > 5 mm in 60 % of episodes"),
    ("pi05_ft", "heavy_lowmu_000", "close", [9.0, 12.5, 15.5, 20.0], (8.0, 22.0),
     "π0.5 fine-tuned · heavy + slippery (2.35 kg, μ 0.5)",
     "pulls it out, then the book rotates (39°) and slips out of the\nfixed 67 N grip during the lift",
     "13/30: 12 of the 17 failures slip in the lift"),
    ("expert", "heavy_lowmu_000", "close", [6.5, 9.0, 11.1, 16.0], (5.0, 16.4),
     "scripted expert · same test case",
     "perfect insertion (0.0 mm), same fixed grip: the book slips too",
     "17/30: even perfect perception loses 13 of 30 with this fixed grip"),
]


def frame_at(frames, t, hz):
    return frames[max(0, min(len(frames) - 1, int(round(t * hz))))]


def fig_outcomes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    n = len(OUTCOME_ROWS)
    fig = plt.figure(figsize=(16, 2.62 * n + 0.5), facecolor=SURFACE)
    import textwrap
    gs = fig.add_gridspec(n, 5, width_ratios=[1.75, 1, 1, 1, 1], wspace=0.03, hspace=0.12,
                          left=0.005, right=0.997, top=1 - 0.5 / (2.62 * n + 0.5), bottom=0.005)
    for r, (pol, ep, view, times, _, title, what, cell) in enumerate(OUTCOME_ROWS):
        fr = read_frames(pol, ep, view)
        hz = media_fps(pol)
        lab = fig.add_subplot(gs[r, 0])
        lab.axis("off")
        lab.add_patch(plt.Rectangle((0.0, 0.08), 0.022, 0.84, transform=lab.transAxes, color=COLOR[pol], clip_on=False))
        lab.text(0.05, 0.94, "\n".join(textwrap.wrap(title, 44)), fontsize=11.5, fontweight="bold", va="top",
                 transform=lab.transAxes, linespacing=1.25)
        lab.text(0.05, 0.62, "\n".join(textwrap.wrap(what.replace("\n", " "), 56)), fontsize=9.6, color=TEXT2,
                 va="top", transform=lab.transAxes, linespacing=1.35)
        lab.text(0.05, 0.2, "\n".join(textwrap.wrap("→ " + cell, 50)), fontsize=10, color=TEXT, va="top",
                 transform=lab.transAxes, fontweight="bold", linespacing=1.25)
        for c, t in enumerate(times):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(frame_at(fr, t, hz))
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(0.02, 0.96, f"t = {t:.1f} s", transform=ax.transAxes, fontsize=9, va="top", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.55), ec="none"))
    fig.suptitle("Why each row of the results table scores what it does: frames of re-runs of evaluation test cases "
                 "(views: whole cell, 3/4 close-up, top of the shelf; the gripper enters from the bottom in the top view)",
                 fontsize=11, color=TEXT2, x=0.005, ha="left", y=0.999, va="top")
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"fig_outcomes.{ext}", dpi=100, facecolor=SURFACE)
    print("wrote", RES / "fig_outcomes.png")


def gif_from_mp4(src: Path, dst: Path, width: int = 720, fps: int = 5, colors: int = 128):
    """README-sized GIF: one shared palette (ffmpeg palettegen/paletteuse), reduced frame rate and width."""
    import subprocess
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-loop", "0", str(dst)], check=True)


def outcomes_clip(seconds: float = 8.0, fps: int = 10, tile=(480, 360)):
    """3 x 2 grid of the rows above, each time-lapsed to `seconds`, labelled; MP4 + a README-sized GIF."""
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()
    tw, th = tile
    bar = 44
    nf = int(seconds * fps)
    tiles = []
    for pol, ep, view, _, (t0, t1), title, what, cell in OUTCOME_ROWS:
        fr = read_frames(pol, ep, view)
        hz = media_fps(pol)
        speed = (t1 - t0) / seconds
        seq = []
        for k in range(nf + fps):           # + 1 s hold on the last frame
            t = min(t1, t0 + (t1 - t0) * k / nf)
            im = Image.fromarray(frame_at(fr, t, hz)).resize((tw, th), Image.LANCZOS)
            canvas = Image.new("RGB", (tw, th + bar), SURFACE)
            canvas.paste(im, (0, bar))
            d = ImageDraw.Draw(canvas)
            d.rectangle([0, 0, 6, bar], fill=COLOR[pol])
            d.text((12, 4), title, font=font, fill=TEXT)
            d.text((12, 24), f"{cell}   ({speed:.1f}x)", font=small, fill=TEXT2)
            d.text((tw - 8, bar + 6), f"t = {t:4.1f} s", font=small, fill="white", anchor="ra")
            seq.append(np.asarray(canvas))
        tiles.append(seq)
    cols = 3
    frames = []
    for k in range(len(tiles[0])):
        rows = [np.concatenate([tiles[r * cols + c][k] for c in range(cols)], axis=1) for r in range(len(tiles) // cols)]
        frames.append(np.concatenate(rows, axis=0))
    out = RES / "media"
    out.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(out / "outcomes.mp4", frames, fps=fps, quality=7, macro_block_size=8)
    gif_from_mp4(out / "outcomes.mp4", out / "outcomes.gif", width=720, fps=5)
    print(f"wrote {out / 'outcomes.mp4'} and .gif ({len(frames)} frames)")


def reel_timing():
    """Clamp whisper word starts to their own line, so a caption never appears before its scene's cut."""
    V = json.loads((REEL / "voice.json").read_text())
    n = 0
    for L in V["lines"]:
        for w in L["words"]:
            if w["t0"] < L["t0"]:
                w["t0"], w["t1"], n = L["t0"], max(w["t1"], L["t0"] + 0.05), n + 1
    (REEL / "voice.json").write_text(json.dumps(V, indent=1))
    print(f"clamped {n} word start(s)")


def reel_publish(h: int = 400):
    """After `reel export`: a 720 x 1280 copy of renders/final.mp4 for the repository and a contact strip of it (one
    frame near the end of every voice line, in two rows) for the README."""
    import subprocess
    from PIL import Image
    mp4 = RES / "media" / "reel_results_60s.mp4"
    times = [round(L["t1"] - 0.25, 2) for L in json.loads((REEL / "voice.json").read_text())["lines"]]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(REEL / "renders" / "final.mp4"), "-vf", "scale=720:1280:flags=lanczos",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(mp4)], check=True)
    tiles = []
    for t in times:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(mp4), "-frames:v", "1", "-f", "image2pipe",
                              "-vcodec", "png", "-"], check=True, capture_output=True).stdout
        import io
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        tiles.append(im.resize((round(im.width * h / im.height), h), Image.LANCZOS))
    gap, per_row = 6, -(-len(tiles) // 2)          # two rows, so each frame stays readable in the README
    tw = tiles[0].width
    strip = Image.new("RGB", (per_row * tw + gap * (per_row - 1), 2 * h + gap), SURFACE)
    for k, t in enumerate(tiles):
        strip.paste(t, ((k % per_row) * (tw + gap), (k // per_row) * (h + gap)))
    strip.save(RES / "media" / "reel_results_60s_strip.jpg", quality=86, optimize=True)
    print("wrote", RES / "media" / "reel_results_60s_strip.jpg", strip.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["suites", "modes", "reel-assets", "reel-timing", "reel-publish"])
    args = ap.parse_args()
    {"suites": fig_suites, "modes": lambda: (fig_outcomes(), outcomes_clip()), "reel-assets": reel_assets,
     "reel-timing": reel_timing, "reel-publish": reel_publish}[args.what]()


if __name__ == "__main__":
    main()
