"""Convert the expert demos (baseline/data/demos/*.npz) to a LeRobot dataset for openpi (openpi's venv).

    HF_LEROBOT_HOME=baseline/data/lerobot $PY -m baseline.openpi_ext.convert_to_lerobot --demos baseline/data/demos

Features: ext_image / wrist_image (224 x 224 RGB), state = [6 joints, gripper], actions = [6 joint targets, gripper
request] (absolute; openpi converts the joints to deltas), task = the instruction. The PDMS skin maps and the wrist
F/T are kept as extra features (tactile_left/right: 7 x 5 x 3 = normal + 2 shear per taxel; tcp_force: 6) for the
tactile follow-up; the vision-only baseline does not read them.
"""
import argparse
import json
import pathlib
import shutil

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

from .configs import REPO_ID

INSTRUCTION_TEMPLATES = ["pick up the {c} book", "pull the {c} book out of the shelf", "take the {c} book",
                         "grab the {c} book from the shelf"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demos", type=pathlib.Path, required=True)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--paraphrase", action="store_true", help="train-time instruction paraphrases (not used by v1)")
    args = ap.parse_args()
    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        shutil.rmtree(out)
    files = sorted(args.demos.glob("*.npz"))
    first = np.load(files[0])
    H, W = first["image_ext_cam"].shape[1:3]
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, robot_type="ur10e_2f140_dohyun_v1", fps=10,
        features={
            "ext_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "tcp_force": {"dtype": "float32", "shape": (6,), "names": ["tcp_force"]},
            "tactile_left": {"dtype": "float32", "shape": (7, 5, 3), "names": ["y", "x", "c"]},
            "tactile_right": {"dtype": "float32", "shape": (7, 5, 3), "names": ["y", "x", "c"]},
        },
        image_writer_threads=16, image_writer_processes=8,
    )
    rng = np.random.default_rng(0)
    n_frames = 0
    for i, f in enumerate(files):
        with np.load(f) as z:          # an NpzFile decompresses on every access: read each array once
            ep = {k: z[k] for k in z.files}
        instr = str(ep["instruction"])
        if args.paraphrase:
            color = json.loads(str(ep["variation"]))["color"]
            instr = INSTRUCTION_TEMPLATES[rng.integers(len(INSTRUCTION_TEMPLATES))].format(c=color)
        state = np.concatenate([ep["joints"], ep["gripper"]], axis=1).astype(np.float32)
        for k in range(len(state)):
            ds.add_frame({
                "ext_image": ep["image_ext_cam"][k], "wrist_image": ep["image_wrist_cam"][k],
                "state": state[k], "actions": ep["actions"][k].astype(np.float32),
                "tcp_force": ep["tcp_force"][k].astype(np.float32),
                "tactile_left": ep["tactile_left"][k], "tactile_right": ep["tactile_right"][k],
                "task": instr,
            })
        ds.save_episode()
        n_frames += len(state)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)} episodes, {n_frames} frames", flush=True)
    print(f"wrote {len(files)} episodes, {n_frames} frames -> {out}")


if __name__ == "__main__":
    main()
