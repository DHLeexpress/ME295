Search below CAD files and put it in this folder (model)
and assembly goes to (\assembly)

model DONE? (Y/n) -> Y (AGC tip kept in model/agc_tip_140 as reference only: not in the assembly, its lug hole fixes the blade mount)
The gripper assembly
- UR10e
- ROBOTIQ gripper 2F-140
- with AGC-TIP-MLD-KIT-140 (NBR rubber material)
- with finally attached gripper_dohyun_v1.stl (version can be changed)

assembly DONE? (Y/n) -> Y: assembly/ur10e_2f140.xml, blade bolted through the shared AGC lug hole (see assembly/README.md)
Note that the final assembly should looks like
/home/dohyun/projects/ME295/model/Screenshot 2026-10-27 at 1.45.37 PM.png

MuJoCo (Important) DONE? (Y/n) -> Y: python scripts_sim/sim_gripper.py (open/close, contact, snapshots in docs/figures/sim_*.png)
Call assembly and get the snapshot out of it
The gripper assembly must be able to open/closed and the contact with external materials must be well defined. 

Note that Jihoon said even if you fully close the gripper, there must be a ~ 1cm gap between them (at least for v1) . -> measured with the shared-hole mount: fully closed, closest blade surfaces 16.8 mm apart, PDMS faces 32.2 mm apart Let me know, including this one, what did you observed and made an assumption on it. Let me know if you need me to download some cad files.
PDMS (2026-10-27) -> Y: the 0.5 mm extrusion on the blade's flat face is the PDMS layer (orange), no extra box;
bonded back face, free front face, friction/stiffness are parameters of scripts_sim/make_blade_assets.py.
Objects are held by the PDMS (python scripts_sim/sim_gripper.py); figures: docs/figures/sim_*.png.
