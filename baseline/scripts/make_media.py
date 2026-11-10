"""Side-by-side clip of the same test episode under each policy (third-person | wrist camera per row).

    python baseline/scripts/make_media.py --suite nominal --index 0
    -> baseline/results/media/compare_<suite>_<index>.mp4 and .gif (GIF: half resolution, every 3rd frame)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "baseline" / "results" / "eval"
ROWS = [("pi05_droid", "π0.5-DROID, zero-shot"), ("pi05_ft", "π0.5 fine-tuned (LoRA)"),
        ("expert", "scripted expert (privileged)")]


def find(policy: str, suite: str, index: int):
    dirs = sorted((d for d in EVAL.iterdir() if d.is_dir() and (d.name == policy or d.name.startswith(policy + "_step"))),
                  key=lambda d: int(d.name.rsplit("_step", 1)[1]) if "_step" in d.name else -1)
    for d in reversed(dirs):
        f = d / f"{suite}.jsonl"
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                if r["index"] == index and r.get("video"):
                    p = Path(r["video"])
                    return (p if p.is_absolute() else ROOT / p), r
    return None, None


def label(frame: np.ndarray, text: str, h: int = 26) -> np.ndarray:
    img = Image.new("RGB", (frame.shape[1], h), (252, 252, 251))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((8, 4), text, fill=(11, 11, 11), font=font)
    return np.concatenate([np.asarray(img), frame], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", default="nominal")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    clips = []
    for pol, name in ROWS:
        path, r = find(pol, args.suite, args.index)
        if path is None:
            continue
        frames = [f for f in iio.mimread(path, memtest=False)]
        if pol == "pi05_droid":                      # recorded at 15 Hz: resample to 10 Hz
            frames = [frames[int(i * 1.5)] for i in range(int(len(frames) / 1.5))]
        outcome = "success" if r["success"] else ("protective stop" if r.get("protective_stop") else "failed")
        clips.append([label(f, f"{name}: {outcome}") for f in frames])
    if not clips:
        raise SystemExit("no videos found")
    n = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (n - len(c)) for c in clips]            # freeze the last frame of shorter episodes
    frames = [np.concatenate([c[i] for c in clips], axis=0) for i in range(n)]
    out = ROOT / "baseline" / "results" / "media"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"compare_{args.suite}_{args.index:03d}"
    iio.mimwrite(f"{stem}.mp4", frames, fps=args.fps, quality=7, macro_block_size=8)
    small = [np.asarray(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2))) for f in frames[::3]]
    iio.mimwrite(f"{stem}.gif", small, duration=0.3, loop=0)          # README-sized: every 3rd frame, half size
    print(f"{stem}.mp4 / .gif: {len(clips)} policies, {n} frames")


if __name__ == "__main__":
    main()
