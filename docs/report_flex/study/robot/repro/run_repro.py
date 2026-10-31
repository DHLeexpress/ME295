"""Stand-alone repro: python run_repro.py [xml]. Steps 0.1 s with islands on and off; prints first step with BADQACC."""
import sys, mujoco
from pathlib import Path
xml = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("minimal_repro.xml"))
for island in (True, False):
    m = mujoco.MjModel.from_xml_path(xml)
    if not island:
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
    d = mujoco.MjData(m)
    bad = None
    for k in range(200):
        mujoco.mj_step(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            bad = k + 1
            break
    print(f"mujoco {mujoco.__version__} islands {'on ' if island else 'off'}: nv {m.nv}, first BADQACC at step {bad}")
