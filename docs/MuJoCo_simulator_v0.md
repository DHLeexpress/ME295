# MuJoCo simulator v0: how to use it

> **v0 — a working draft, not the final simulator.** It runs the full librarian task and the skin ablation, but the
> skin model, the visuals and the controller tuning are still rough (see [Known limits of v0](#known-limits-of-v0)).
> Interfaces may change in v1. Deeper background: [SIM.md](SIM.md) (how MuJoCo computes force, the controller slot)
> and the [flex study](report_flex/flex_study.pdf) (how far the skin model can be trusted).

**What it is.** A MuJoCo 3.17 model of our cell: UR10e + Robotiq 2F-140 with the custom blade fingers
(`gripper_dohyun_v1`), each carrying the 51 × 71 × 1.5 mm PDMS tactile skin, in a desk scene with two bookshelves. The
robot is driven through the **python-urx API**, so the same task script is meant to run on the real UR10e.

## 1. Install

```bash
git clone https://github.com/DHLeexpress/ME295.git && cd ME295
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # numpy, matplotlib, mujoco>=3.17, trimesh, imageio-ffmpeg
python -m unittest discover -s tests                    # ~15 s, includes the simulator tests
```

Python 3.10–3.12. Headless Linux with a GPU: the scripts set `MUJOCO_GL=egl` themselves when there is no display.

## 2. Run it

| I want to… | Command |
|---|---|
| browse the scene (drag to rotate, double-click + Ctrl-drag to push a body) | `python -m mujoco.viewer --mjcf assembly/scene_bookshelf.xml` |
| watch the librarian task live with a HUD | `python scripts_sim/run_librarian.py --viewer` (macOS: `mjpython …`) |
| …in slow motion | add `--realtime 0.25` |
| run it headless (log, CSV and summary plot in `results/sim_runs/librarian/`) | `python scripts_sim/run_librarian.py` |
| make the video (3D skin panel, MuJoCo views, wrist camera) | `python scripts_sim/run_librarian.py --video results/librarian.mp4` |
| …with the old 2D skin maps | add `--panel 2d` |
| run without the skin (constant grip, register 0–255 = 10–125 N per finger) | `python scripts_sim/run_librarian.py --controller fixed --grip-force 128` |
| skin vs fixed grip over book mass and friction (40 runs in parallel) | `python scripts_sim/ablation.py` (`--quick` for a short check) |
| open/close the gripper, grasp a box, save snapshots | `python scripts_sim/sim_gripper.py` |
| run the same task on the real UR10e (asks for a typed "yes"; untested on hardware) | `python scripts_sim/run_librarian.py --backend urx --host <robot ip>` |

In the viewer: `C` contact points, `F` contact forces, `Tab` panels; cameras `cam_main`, `cam_close`, `cam_side`,
`wrist_cam` under *Rendering → Camera*. Video renderer: `--renderer auto|filament|classic` (Filament needs Vulkan).

**The task** (`scripts_sim/run_librarian.py`): approach shelf A → slide both blades into the ~10 mm gaps beside a 40 mm
book (stops if the wrist F/T exceeds 15 N) → squeeze with the tactile controller → pull out 21 cm → lift → carry →
slide into the 70 mm slot of shelf B (stops on an unexpected 10 N) → set down, release, retreat. Success = the book's
centre within 15 mm horizontally / 8 mm vertically of the slot, tilt and yaw below 5°. One run: ~37 s simulated,
~2 min wall.

## 3. Use it from Python

```python
import sys; sys.path.insert(0, "src")
from shear_agent.robot import SimRobot, SimRobotiqGripper, PDMSSkin

rob = SimRobot("assembly/scene_bookshelf.xml", verbose=False)   # urx-compatible UR10e in MuJoCo
gripper = SimRobotiqGripper(rob)
skin = PDMSSkin(rob.m, "left")                                   # 5 x 7 taxels on the left blade

pose = rob.getl()                       # [x, y, z, rx, ry, rz]: base frame, metres, rotation vector
pose[2] += 0.05
rob.movel(pose, acc=0.5, vel=0.1)       # blocks until done (in simulated time)
gripper.gripper_action(200)             # 0 open .. 255 closed
rob.sleep(0.5)                          # advance simulated time
frame = skin.read(rob.d)                # TactileFrame: normal (7,5) N, shear (7,5,2) N, deformation (7,5,3) m
print(frame.total_normal, rob.get_tcp_force())
```

Also available: `getj`, `movej`, `speedl`, `stopl`, `get_tcp_force`, the Robotiq registers (`set_position`,
`set_force`, `set_speed`). `rob.m` / `rob.d` are the raw `mujoco.MjModel` / `MjData`. To plug in your own controller
(MPC, MPPI, a learned policy) implement `Controller.act(obs) -> Command` in `src/shear_agent/control/` — see
[SIM.md §4](SIM.md#4-where-the-controller-goes).

## 4. Change the PDMS skin

The skin files `assembly/pdms_{left,right}.xml` and the matching solver settings `assembly/pdms_option.xml` are
generated — edit the generator's options, not the XML:

```bash
# default: 5 x 7 Winkler cells (fast; used by the demo)
python scripts_sim/make_blade_assets.py model/gripper_dohyun_v1.stl --pdms-only --pdms-model cells
# finite-element flex (slow on the robot; see the flex study before using it)
python scripts_sim/make_blade_assets.py model/gripper_dohyun_v1.stl --pdms-only --pdms-model flex --flex-grid 6x8x2
```

| option | default | meaning |
|---|---|---|
| `--pdms-model` | `cells` | `cells`: 5 × 7 spring columns (Winkler); `flex`: MuJoCo tetrahedral FEM |
| `--thickness` | 1.5 mm | layer thickness (0.5 mm proud of the blade, the rest in a pocket) |
| `--young`, `--poisson` | 1 MPa, 0.45 | placeholders (real PDMS: ν ≈ 0.49–0.5); for the flex the generator corrects MuJoCo's doubled λ automatically |
| `--tau` | 0 | damping time constant (0 = linear elastic, no damping) |
| `--cells` / `--flex-grid` | `5x7` / `6x8x2` | cell grid / flex vertices (more than 3 layers through the thickness makes the flex contact *softer*) |
| `--mu` | 1.0 | skin friction |

Run the unit tests afterwards. Switching the model also switches the solver: Newton for the cells; CG with 1000
iterations and islands disabled for the flex (both required by MuJoCo 3.17 defects we found).

## 5. Where things are

| path | what |
|---|---|
| `assembly/scene_bookshelf.xml` | the librarian scene (includes the robot `ur10e_2f140.xml`) |
| `assembly/scene.xml` | a simple scene with a test box on a pedestal |
| `scripts_sim/` | task (`run_librarian.py`), video and panels (`librarian_media.py`, `panel3d.py`), ablation, generators |
| `src/shear_agent/robot/` | `SimRobot`, `SimRobotiqGripper`, skin readout (`tactile.py`), renderer, real-robot adapter |
| `src/shear_agent/control/` | controller interface; skin controller `ShearMarginGrasp`, baseline `FixedForceGrasp` |
| `results/ablation_rev/` | the latest ablation (40 runs) |
| `docs/media/` | demo and skin close-up videos |

## Known limits of v0

- **Skin model.** The default 5 × 7 cells are right for whole-face loads at the placeholder ν = 0.45 but 1.2–30× too
  stiff for taxel-sized loads, especially at real PDMS's ν ≈ 0.5. The FEM flex is accurate in elasticity but its
  contact halves the torsional grip, the wrist F/T does not see its forces, and it is ~15× slower on the robot.
- **Sensor.** The panels show what the model computes on a 5 × 7 grid at an assumed 20 Hz, without noise or
  calibration against the real skin.
- **Task.** Scripted, with known shelf and book poses (no perception) and one book geometry; no LLM agent in the
  loop yet. The friction-ratio margin cannot detect slip when μ is overestimated.
- **Hardware.** The urx backend has not been run on the real UR10e from this repo.
- **Looks.** Renders, panels and video layout are functional, not polished.
