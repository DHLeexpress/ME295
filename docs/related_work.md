# Related-work note (Lab Report 1)

**Question we read for:** what has already been tried for (A) tactile grip-force control and slip, (B) pulling books or thin objects out of packed shelves, and (C) agents or policies that use touch, and where is the gap for a *thin, shear-sensing parallel-jaw finger*?

BibTeX for everything below: [`refs.bib`](refs.bib).

## Three core papers

### 1. Trixi the Librarian (Wieczorek, Liu, Sygo, Koshil; Univ. Hamburg TAMS; arXiv 2210.10110, 2022)
- **What it does.** A PR-2 sorts books on a shelf: YOLOv5 spine detection + colour-histogram matching → a simple task planner → bimanual manipulation with MoveIt/BioIK.
- **How it gets a book out.** One Shadow-hand finger presses on the top of the book and tilts it out. The other arm's two-finger gripper grasps the exposed spine, pulls a little, releases so the book falls back with more spine exposed, regrasps, and pulls it out. A force-torque threshold is used only when placing.
- **What went wrong.** Tilt-and-grasp with the Shadow hand alone failed (workspace), a hook-shaped finger collided with the shelf above, and the method assumes aligned, upright spines. The demo sorts three books.
- **What we take.** This is the canonical **baseline for B**: scripted tilt → grasp → pull with no contact sensing, which needed two arms and a dexterous hand. Our question is whether *one* parallel-jaw gripper with thin shear-sensing fingers can tilt, pinch and pull while *feeling* tilt onset and slip.

### 2. DeliGrasp (Xie, Valentini, Lavering, Correll; CoRL 2024, PMLR 270; arXiv 2403.07832)
- **What it does.** An LLM infers an object's mass, friction coefficient and spring constant from a text description, then writes a first-principles adaptive grasp policy for a two-finger gripper whose torque is set by limiting motor current (UR5 + MAGPIE gripper).
- **Result.** LLM-parameterized physics policies beat both traditional adaptive grasping and direct LLM-as-code policies on 12 delicate or deformable items (2–900 g).
- **What we take.** This is the closest **agentic baseline for A and C**. Its physical parameters are *priors from language*, and the only contact feedback is motor current/position along the closing direction. Shear is never measured, so μ and the slip margin cannot be checked online, which is exactly what our sensor adds. The natural experiment: DeliGrasp-style prior → our shear-margin feedback.

### 3. Reactive Diffusion Policy, RDP (Xue et al.; RSS 2025; arXiv 2503.02881)
- **What it does.** Slow-fast visuo-tactile imitation learning. A slow latent diffusion policy predicts action chunks at low rate, and a fast asymmetric tokenizer closes the loop on tactile/force feedback inside each chunk. TactAR (AR teleoperation with tactile feedback) collects the demos.
- **Setup and result.** Flexiv arms with GelSight Mini (25 fps), MCTac or joint-torque sensing, on peeling, wiping and bimanual lifting. It beats vision-only imitation-learning baselines.
- **What we take.** This is the **scaled-up, learned comparator**. It shows touch helps learned policies, but it needs teleop demonstrations for every task and fast optical fingertips. Our ~20 Hz shear sensing plus a physics-based margin is a low-data alternative. An RDP-style visuo-tactile policy on our sensor is a stretch comparison.

## Adjacent work (one line each)
- **Ford et al. 2025** (arXiv 2503.17501): shear-based grasp control on an underactuated SoftHand with TacTip-type fingertips, using simple controllers. This is the classical shear-feedback design we adapt for a parallel jaw.
- **Dong, Yuan, Adelson, IROS 2017** (arXiv 1708.00922): GelSight slip detection triggers a regrasp with more force (88 of 99 grasps succeeded). This is our **slip-reactive baseline**.
- **Dong & Rodriguez, IROS 2019** (arXiv 1909.05426): GelSlim feedback for inserting objects into densely packed boxes. It is tactile manipulation in tight gaps, done as insertion rather than extraction.
- **Zhou & Held, CoRL 2022** (arXiv 2211.01500): an RL policy for a parallel gripper pushes objects against a wall to grasp them (extrinsic dexterity), with no touch sensing.
- **Zheng et al. 2026** (arXiv 2608.26883): an active-surface gripper pulls books off a packed shelf by spinning a high-friction surface. This is the *hardware* route to application B.
- **Jones et al., ICRA 2025, FuSe** (arXiv 2501.04693): fine-tunes generalist policies (Octo, a PaliGemma VLA) with touch and audio via language grounding.
- **Bi et al. 2025, VLA-Touch** (arXiv 2507.17294): a tactile-language model for replanning plus tactile refinement of a frozen VLA's actions.
- **Higuera et al., CoRL 2024, Sparsh**: self-supervised representations for *vision-based* tactile sensors. Transfer to our skin's signals is unclear.
- **Yu et al., RSS 2024, Octopi**: a tactile-language model that reasons about physical properties from GelSight videos.

## Synthesis: where the gap is
1. **Sensing.** Shear-feedback grasping exists for camera-based fingertips (GelSight, TacTip, GelSlim), which are thick. It has not been shown with a *thin tactile skin on an industrial parallel-jaw gripper*. There, normal force is already known from the gripper and shear is the one missing variable in the friction-cone margin.
2. **Book extraction.** Prior systems use a dexterous hand plus a second arm (Trixi), special hardware (active surface, suction) or no contact sensing at all. Thin fingers that *feel* tilt onset, slip and jamming against neighbouring books are untested.
3. **Agents.** LLM grasping sets physical parameters open-loop from language (DeliGrasp), and LLM monitors mostly watch execution through vision. Touch is rarely exposed to an agent as callable tools. Learned visuo-tactile policies (RDP, FuSe, VLA-Touch) need per-task demos and optical sensors.

**Our niche:** a physics-grounded shear margin (A), applied to a confined, occluded task where vision is weak (B), with touch offered to an LLM agent as a tool (C). Each layer has a concrete baseline from the papers above.

| Layer | Baseline(s) we will run | From |
|---|---|---|
| A | fixed force; slip-reactive regrasp; LLM-prior force (no touch) | gripper default; Dong et al. 2017; DeliGrasp |
| B | scripted vision-only tilt → grasp → pull | Trixi |
| C | same agent with vision-only tools | ablation |
| learned (stretch) | vision-only Diffusion Policy vs RDP-style visuo-tactile policy on ~50 demos | RDP |
