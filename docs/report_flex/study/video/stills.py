"""stills.png: one composite frame per object (end of the hold, i.e. full normal load before the slide),
2 columns x 4 rows at half resolution (1920 x 2160)."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, os.environ.get("RUNS", "prod2"))
FR = os.path.join(HERE, os.environ.get("FRAMES", "frames"))
tiles = []
for k in range(8):
    z = np.load(os.path.join(RUNS, f"seg{k}.npz"), allow_pickle=True)
    hold = np.where(z["phase"] == 2)[0][-1]
    fi = int(np.searchsorted(z["f_step"], hold, side="right") - 1)
    im = plt.imread(os.path.join(FR, f"seg{k}_{fi:04d}.png"))[..., :3]
    im = im[: im.shape[0] // 2 * 2, : im.shape[1] // 2 * 2]
    tiles.append(0.25 * (im[::2, ::2] + im[1::2, ::2] + im[::2, 1::2] + im[1::2, 1::2]))
    print(k, "frame", fi)
rows = [np.concatenate(tiles[2 * r:2 * r + 2], axis=1) for r in range(4)]
plt.imsave(os.path.join(HERE, "stills.png"), np.clip(np.concatenate(rows, axis=0), 0, 1))
