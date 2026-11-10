# Top-down grasp ablation (simulated tabletop, wrist ZED perception)

Success = both pads on the object and lifted > 5 cm. Same perception, gripper sequence and safety filter for every controller. 95 % Wilson intervals.

| controller | nominal | calib | calib2 | moved | fatten |
|---|---|---|---|---|---|
| scripted | 30/30 [89, 100 %] | 30/30 [89, 100 %] | 30/30 [89, 100 %] | 18/30 [42, 75 %] | 30/30 [89, 100 %] |
| pbvs | 30/30 [89, 100 %] | 30/30 [89, 100 %] | 30/30 [89, 100 %] | 30/30 [89, 100 %] | 30/30 [89, 100 %] |
| mpc | 30/30 [89, 100 %] | 30/30 [89, 100 %] | 30/30 [89, 100 %] | 30/30 [89, 100 %] | 30/30 [89, 100 %] |

Medians over all episodes of a controller (time: start to the end of the hold):

| controller | time s | goal error mm | pushed before closing mm | gripper-body contact N | safety interventions | perceptions |
|---|---|---|---|---|---|---|
| scripted | 9.0 | 7.2 | 3.1 | 3.9 | 0.0 | 1.0 |
| pbvs | 9.8 | 2.0 | 1.2 | 0.3 | 0.0 | 19.0 |
| mpc | 9.6 | 1.9 | 1.2 | 0.1 | 0.0 | 24.0 |

Failures:

- scripted / moved #2: missed (jaws closed on nothing); cylinder 74 mm, lift -749 mm, pushed 1509 mm, goal error 42.0 mm
- scripted / moved #3: missed (jaws closed on nothing); cylinder 51 mm, lift -61 mm, pushed 16 mm, goal error 38.2 mm
- scripted / moved #8: missed (jaws closed on nothing); sphere 89 mm, lift -0 mm, pushed 107 mm, goal error 38.9 mm
- scripted / moved #11: missed (jaws closed on nothing); sphere 75 mm, lift -0 mm, pushed 135 mm, goal error 40.5 mm
- scripted / moved #16: done; cylinder 59 mm, lift -0 mm, pushed 18 mm, goal error 39.2 mm
- scripted / moved #17: missed (jaws closed on nothing); box 54 mm, lift -53 mm, pushed 111 mm, goal error 39.2 mm
- scripted / moved #20: done; cylinder 56 mm, lift 104 mm, pushed 5 mm, goal error 41.6 mm
- scripted / moved #21: done; sphere 61 mm, lift -0 mm, pushed 38 mm, goal error 38.6 mm
- scripted / moved #24: missed (jaws closed on nothing); cylinder 51 mm, lift -39 mm, pushed 99 mm, goal error 41.9 mm
- scripted / moved #25: done; sphere 88 mm, lift -0 mm, pushed 63 mm, goal error 37.9 mm
- scripted / moved #28: done; cylinder 52 mm, lift 103 mm, pushed 5 mm, goal error 40.6 mm
- scripted / moved #29: done; cylinder 69 mm, lift -0 mm, pushed 26 mm, goal error 40.5 mm
