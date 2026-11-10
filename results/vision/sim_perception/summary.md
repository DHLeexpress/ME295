# Wrist ZED shelf perception vs simulation ground truth

Look pose aimed at the row centre (target unknown), one frame, HD720 intrinsics of the calibrated ZED Mini.
Errors in mm (estimate - truth): median [5th, 95th percentile]. found = the instruction's book picked.

| suite | depth | n | found | wrong book | no candidate | centre x | thickness | gap left | gap right | spine plane | top |
|---|---|---|---|---|---|---|---|---|---|---|---|
| narrow | exact | 30 | 30 | 0 | 0 | -0.1 [-0.9, +0.7] | +0.8 [-0.1, +1.9] | -1.1 [-2.0, +0.1] | -1.1 [-2.6, -0.1] | +0.0 [-0.0, +0.0] | -2.3 [-2.7, -2.2] |
| narrow | noise | 30 | 30 | 0 | 0 | -0.1 [-0.9, +0.7] | +0.8 [-0.1, +1.6] | -1.1 [-2.0, +0.1] | -1.1 [-2.6, -0.1] | -0.0 [-0.0, +0.0] | -2.3 [-2.7, -2.1] |
| narrow | fatten | 30 | 9 | 3 | 18 | -0.1 [-0.5, +0.4] | +2.4 [+2.1, +2.6] | -3.0 [-3.4, -2.3] | -2.9 [-3.3, -2.0] | -0.3 [-0.4, -0.3] | -1.2 [-1.4, -1.0] |
| nominal | exact | 30 | 30 | 0 | 0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.4 [-0.4, -0.4] | -0.4 [-0.4, -0.4] | -0.0 [-0.0, -0.0] | -2.6 [-2.6, -2.6] |
| nominal | noise | 30 | 30 | 0 | 0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.4 [-0.4, -0.4] | -0.4 [-0.4, -0.4] | -0.0 [-0.0, -0.0] | -2.6 [-2.6, -2.6] |
| nominal | fatten | 30 | 30 | 0 | 0 | +0.0 [+0.0, +0.0] | +1.6 [+1.6, +1.6] | -2.0 [-2.0, -2.0] | -2.8 [-2.8, -2.8] | -0.4 [-0.4, -0.3] | -1.4 [-1.4, -1.4] |
| random | exact | 30 | 30 | 0 | 0 | -0.2 [-0.9, +0.6] | +1.1 [-0.2, +1.6] | -1.4 [-2.3, -0.3] | -1.5 [-2.2, +0.0] | -0.0 [-0.0, +0.0] | -2.5 [-2.7, -2.1] |
| random | noise | 30 | 30 | 0 | 0 | -0.2 [-1.1, +0.7] | +1.4 [-0.2, +1.6] | -1.4 [-2.3, -0.3] | -1.6 [-2.2, +0.0] | -0.0 [-0.0, +0.0] | -2.5 [-2.7, -2.2] |
| random | fatten | 30 | 29 | 0 | 1 | -0.3 [-0.9, +0.8] | +3.2 [+2.3, +4.0] | -3.6 [-4.0, -2.7] | -3.5 [-4.8, -2.4] | -0.3 [-0.4, -0.3] | -1.2 [-1.5, -1.0] |

Misses:

- narrow #0 (fatten): no candidate (asked black)
- narrow #1 (fatten): no candidate (asked orange, distractor present)
- narrow #2 (fatten): wrong book, picked red (asked orange, distractor present)
- narrow #4 (fatten): no candidate (asked purple)
- narrow #9 (fatten): no candidate (asked purple, distractor present)
- narrow #10 (fatten): no candidate (asked blue)
- narrow #13 (fatten): wrong book, picked green (asked red, distractor present)
- narrow #14 (fatten): no candidate (asked yellow)
- narrow #15 (fatten): no candidate (asked red)
- narrow #16 (fatten): no candidate (asked green)
- narrow #18 (fatten): no candidate (asked green)
- narrow #19 (fatten): no candidate (asked green)
- narrow #20 (fatten): no candidate (asked yellow)
- narrow #22 (fatten): no candidate (asked green)
- narrow #23 (fatten): no candidate (asked black)
- narrow #24 (fatten): no candidate (asked black)
- narrow #25 (fatten): no candidate (asked white)
- narrow #26 (fatten): no candidate (asked purple)
- narrow #27 (fatten): wrong book, picked red (asked green, distractor present)
- narrow #28 (fatten): no candidate (asked purple)
- narrow #29 (fatten): no candidate (asked orange)
- random #4 (fatten): no candidate (asked green)
