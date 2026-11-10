# Calibration check 2026-11-09: keep the active calibration

**Session.** 15 captures (14 distinct poses: 14 and 15 are identical). Board on the wall shelf, 0.84–1.28 m from the
camera. ZED SDK 5.4, rectified HD720, fx = 726.8 px. Robot TCP = flange (1.7 mm / 0.32° check).

| calibration | 2026-10-13 session (18 poses, board on the table) | this session (15 poses) |
|---|---|---|
| **active, 2026-10-13, unchanged** | 3.4 ± 2.1 mm, 0.28° | **6.1 ± 0.9 mm, 0.32°** |
| joint refinement on both sessions (`candidate_joint_refined_with_20261013.npz`; differs from the active by 2.1 mm / 0.17°) | 3.5 ± 2.4 mm, 0.30° | 5.0 ± 1.6 mm, 0.29° |
| Tsai on this session alone (`hand_eye_calibration_ur10e_result.npz`; differs by 26 mm / 2.2°) | – | 18.9 ± 11.0 mm, 1.53° |

Consistency is the spread of the board's base-frame pose over the poses (the board does not move): mean ± std of the
distance to the average pose.

## Decision: keep the active calibration
- **It holds on independent data.** Today's data was taken a month later, with the board elsewhere and a new SDK, and
  the active calibration explains it to ~6 mm / 0.3° at ~1.1 m.
- **Re-fitting gains almost nothing.** Using both sessions improves this session by ~1 mm and moves the calibration by
  2 mm / 0.2°. That is within the noise, so the candidate is kept for reference only.
- **Do not use the stand-alone Tsai result.** It is ill-conditioned: the board was far away (an angle error is
  multiplied by the ~1.1 m lever arm) and there were few distinct tilts. All of OpenCV's hand-eye methods agree with it
  (within 1.5 mm / 0.2°), so the data, not the solver, is the limit. Holding out one pose at a time, it predicts the
  board 17 mm off (median).
- **For better than ~4 mm**, capture 15–20 poses with the board flat on the table 0.4–0.7 m from the camera, tilting
  the camera ±20–30° in different directions rather than only spinning the wrist.

## 한국어 요약
- **결론: 현재(2026-10-13) 캘리브레이션을 그대로 씁니다.**
- **새 데이터로 검증됐습니다.** 오늘 데이터는 한 달 뒤에, 보드 위치도 SDK 버전도 다른 상태에서 찍었는데, 현재
  캘리브레이션이 약 6 mm / 0.3° 일관성으로 잘 맞습니다.
- **두 세션을 합쳐 다시 맞춰도 거의 차이가 없습니다.** 약 1 mm 개선으로 노이즈 수준이라, 결과는 참고용으로만
  보관합니다.
- **오늘 데이터만으로 계산한 새 결과는 쓰지 마세요.** 보드가 1 m 이상 멀었고 자세 각도가 다양하지 않아서
  결과가 불안정합니다.
- **더 정확하게(4 mm 이하) 하려면** 보드를 테이블 위 카메라에서 0.4–0.7 m 거리에 놓고, 손목을 돌리기만 하지
  말고 여러 방향으로 ±20–30° 기울이며 15–20개 자세를 캡처하세요.
