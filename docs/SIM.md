# MuJoCo simulation: setup, the librarian demo, and what the numbers mean

The simulated cell is a UR10e with a Robotiq 2F-140, the custom blade fingers `gripper_dohyun_v1` (with their thin PDMS layer), a wrist camera, a desk and a bookshelf. The task code uses the **python-urx API**, so the same script drives the real robot. MuJoCo 3.17 or newer.

Demo video (35 s, 1080p): [media/librarian_demo.mp4](media/librarian_demo.mp4). It has three panels: what the skin feels, the MuJoCo view (Filament) with a tool-following close-up, and the wrist camera with the urx console. The run's force and slip traces:

![librarian run](figures/sim_librarian_summary.png)

## 1. Install (Linux, macOS, Windows)

```bash
git clone https://github.com/DHLeexpress/ME295.git && cd ME295
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # numpy, matplotlib, mujoco>=3.17, trimesh, imageio-ffmpeg
python -m unittest discover -s tests                     # ~10 s, includes the simulator tests
```

Python 3.10 to 3.12. Nothing else is needed for the viewer and the classic renderer. For video, `imageio-ffmpeg` ships an ffmpeg binary.

## 2. Look at it

| Goal | Command |
|---|---|
| Browse the scene (drag to rotate; double-click a body, Ctrl+drag to push it) | `python -m mujoco.viewer --mjcf assembly/scene_bookshelf.xml` |
| Watch the librarian task live, with HUD (joints, TCP, gripper registers, skin forces, taxel maps) | `python scripts_sim/run_librarian.py --viewer` |
| Same on **macOS** (the passive viewer needs MuJoCo's launcher) | `mjpython scripts_sim/run_librarian.py --viewer` |
| Slow motion | add `--realtime 0.25` |
| Headless run: urx-style log, CSV and a summary plot in `results/sim_runs/librarian/` | `python scripts_sim/run_librarian.py` |
| Three-panel video: skin, MuJoCo, wrist camera | `python scripts_sim/run_librarian.py --video results/librarian.mp4` |
| Gripper checks and figures (exploded view, closed section) | `python scripts_sim/sim_gripper.py`, `python scripts_sim/plot_connection.py` |

In the viewer, press `C` for contact points, `F` for contact forces and `Tab` for the UI panels. The fixed cameras (`cam_main`, `cam_close`, `cam_side`, `wrist_cam`) are under *Rendering → Camera*.

### Rendering backends (`--renderer auto|filament|classic`)

- **Filament** (MuJoCo 3.17+): physically based rendering with image-based lighting, soft shadows and MSAA. It uses **Vulkan** and needs no window, so it runs headless on GPU servers, on Linux desktops with NVIDIA/AMD/Intel drivers, and on Windows.
  - Check with `vulkaninfo --summary`.
  - Its own decorations aren't exposed to Python yet, so force arrows and contact markers are drawn by a classic overlay pass.
- **Classic OpenGL**: works wherever MuJoCo does, and `auto` falls back to it.
  - Headless Linux with a GPU: `MUJOCO_GL=egl` (the scripts set this when there is no `DISPLAY`).
  - No GPU: `MUJOCO_GL=osmesa` (`sudo apt install libosmesa6`), slow.
  - macOS / Windows: leave `MUJOCO_GL` unset.
  - macOS has no Vulkan by default, so expect the classic renderer there.

## 3. Same code, real robot

| Simulation (`shear_agent.robot`) | Real UR10e |
|---|---|
| `rob = SimRobot(scene)` | `rob = urx.Robot("192.168.x.x")` |
| `gripper = SimRobotiqGripper(rob, speed, force)` | `RobotiqSocket(host)` (URCap port 63352), or urx's `Robotiq_Two_Finger_Gripper(rob)` |
| `rob.getj()`, `getl()`, `get_pose()`, `get_tcp_force()`, `get_force()` | same |
| `rob.movej(q, acc, vel, wait)`, `movel(pose, acc, vel, wait)`, `translate`, `up/down`, `speedl`, `speedj`, `stopl`, `stopj`, `is_program_running()`, `set_tcp`, `set_payload` | same |
| `gripper.gripper_action(0..255)`, `open_gripper()`, `close_gripper()` | same |
| `gripper.set_position/set_force/set_speed`, `get_position()`, `get_object_status()` | `RobotiqSocket` (non-blocking register access) |
| `rob.sleep(dt)` advances simulated time | `real_ur.connect()` sets `rob.sleep = time.sleep` |

- **Poses** are urx's `[x, y, z, rx, ry, rz]`: base frame, metres, rotation vector.
- **Kinematics** are UR's published UR10e DH parameters. The sim's flange pose equals the controller's to machine precision (`tests/test_sim_robot.py`).
- **Run on the robot:** `python scripts_sim/run_librarian.py --backend urx --host <ip>` (asks for confirmation; `pip install urx` first).
- **Not tested on hardware from this repo.** Before the first real run:
  - verify `set_tcp`/`set_payload` on the pendant;
  - check the sign of `get_tcp_force()` with a known weight;
  - keep the speeds as they are.

## 4. Where the controller goes

The task is gross motion with urx calls, then the contact-rich phase in a controller loop (`shear_agent.control`):

```python
class MyController(Controller):        # MPC, MPPI, VLFA ... (placeholders in control/placeholders.py)
    rate_hz = 50.0
    def act(self, obs: Observation) -> Command:
        # obs: q, tcp_pose, tcp_force, Robotiq registers, tactile {"left","right": TactileFrame}, images
        return Command(tcp_velocity=[vx, vy, vz, wx, wy, wz], gripper_position=255, gripper_force=40)
```

`run(controller, rob, gripper, skins)` turns each Command into `rob.speedl(...)` and Robotiq register writes, so a new controller works in simulation and on the robot. For model-based control, roll the MuJoCo model forward from a copy of `rob.d` (`mujoco.MjData`, `mujoco.rollout`).

The placeholder `ShearMarginGrasp` closes with the lowest force and computes each blade's patch margin q from the skin. It raises the Robotiq force register while q < q\* and pauses the arm while q < q\_pause. It pulls the book out 0.25 m and lifts it 0.08 m. The run then stands the book on the desk with a force-guarded descent.

## 5. How MuJoCo defines force and displacement

**Bodies are rigid; contacts are constraints.** Each time step, MuJoCo finds the contacts between geoms and solves one optimisation for all constraint forces: contacts, joint limits and the finger linkage's equality constraints. Each contact gets a force in its own contact frame:

- a **normal force** (≥ 0, pushing the geoms apart);
- two **friction forces** tangent to the surfaces, inside the friction cone |f_t| ≤ μ f_n (elliptic cone here). μ is the geom's `friction`, and the PDMS taxels' `priority` makes their μ = 1.0 win;
- with `condim=4`, also a **torsional friction** moment.

These are real forces in newtons: they hold the book up against gravity and obey Newton's laws. `mj_contactForce` returns them, and everything below is a sum of them.

| What you see | How it is computed |
|---|---|
| Skin taxel normal / shear (`TactileFrame.normal`, `.shear`, the heat map and arrows) | Each blade's 0.5 mm PDMS layer collides as 5 × 7 taxel boxes. Per taxel: the sum of its contacts' forces, rotated into the skin frame. Normal = into the skin; shear = in-plane drag along x (across the blade) and y (toward the tip). |
| Skin as the real sensor delivers it (`SkinSensor`) | The taxel forces sampled every 2 ms and averaged into 20 Hz frames (the lab pipeline is ~50 ms/frame). Contact forces chatter at hundreds of Hz while the book slides on the shelf; a real sensor wouldn't see that. |
| Patch margin q | 1 − √((\|T\|/μN)² + (\|M\|/μN·r)²): T = summed shear, N = summed normal, M = the shear field's twist about the centre of pressure, r = (2/3)√(A/π) of the loaded area. It falls toward 0 as the patch nears sliding or twisting. |
| Bond sensor `{side}_pdms_force` | MuJoCo `force` sensor: the force the PDMS layer transmits to the blade (a load cell behind the skin). It equals the taxel sum. |
| Arm F/T `rob.get_tcp_force()` | MuJoCo `force`/`torque` sensors at the flange (`attachment_site`), as the UR10e's built-in sensor. Rotated to the base frame, moments taken about the TCP, with the declared payload's weight subtracted (as UR does). Holding the 0.8 kg book reads Fz ≈ −7.8 N. |
| Gripper force | The finger actuator's torque, capped by the force register: 10–125 N per finger (2F-140 data sheet), calibrated in `sim_gripper.py`. |

**Displacement**:

- **Rigid motion:** body positions and orientations (`d.xpos`, `d.xquat`) and joint angles (`d.qpos`). There is no strain inside a body.
  - `getj()`/`getl()` are joint angles and the TCP pose.
  - "Slip" in the plots is the book's displacement relative to the TCP since the grasp closed.
  - "Neighbours moved" is the largest displacement of any other book.
- **Contact penetration:** contacts are **soft**. The solver lets geoms overlap slightly (`contact.dist < 0`), and the force grows with the overlap and its rate.
  - `solref` = (time constant, damping ratio) and `solimp` = impedance set this relation: a regularised spring-damper, not a material law.
  - For the PDMS taxels it is set to give ~75 µm of penetration at ~30 N (`make_blade_assets.py --solref/--solimp`). That stands in for the combined compliance of the PDMS and the book's cover; reported as `TactileFrame.penetration`.
  - Without it, two "perfectly flat" faces 0.02° apart put the whole load on one row of taxels.

**Caveats:**

- **Load sharing:** how load splits among many contacts (35 taxels × 2 blades) is not unique for rigid bodies. MuJoCo picks the solution of its regularised problem, and compliance makes it plausible, so read taxel maps as qualitative. Net forces are exact.
- **Friction model:** no static/kinetic difference, and soft constraints allow slow creep under tangential load.
- **Thin layer:** the 0.5 mm PDMS layer is rigid apart from the contact compliance.
- **Sensor calibration:** the real skin measures a strain field. Calibrating strain against these simulated tractions is the link to the hardware.
