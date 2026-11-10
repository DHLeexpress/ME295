# zed_calibration_live: live check / recapture of the wrist-ZED hand-eye calibration

A side folder next to `vision/zed_calibration/`, which (like the lab's original `../ZED`) is never modified here.

```bash
python vision/zed_calibration_live/live_calibration_view.py          # window: board detection + live calibration check
python vision/zed_calibration_live/live_calibration_view.py --solve vision/zed_calibration_live/captures/<session>
```
- **Read-only towards the robot.** It reads the pose over RTDE and never moves the arm. Move it with the pendant or
  freedrive.
- **Consistency check.** Each time the arm stands still at a new pose (> 2 cm or > 5° from the last sample) with the
  board in view, the board's pose in the base frame, computed through the *active* calibration, is added as a
  sample. The board does not move, so the spread over poses measures the calibration. The original session had
  3.4 ± 2.1 mm.
- **Keys:**
  - `y` captures a pose (image, `T_base_ee`, `T_camw_board`, `q`, in the original `poses.npz` format) into
    `captures/<session>/`.
  - `h` solves Tsai AX = XB on that session and saves `hand_eye_calibration_ur10e_result.npz` + `solve.json` there,
    comparing it with the active calibration.
  - `r` resets the samples; `q` quits.
- **Adopting a new result** is your decision. Point `vision/zed_calibration/calibration/active_calibration.txt` (or
  `$ME295_ZED_DIR`) at it. The tool never does that.
- **Raw vs SDK.** Without the ZED SDK the view uses the raw, unrectified UVC image, so its numbers are approximate:
  good for setting up poses, not for the final calibration. With the SDK (`pyzed`) it uses the rectified image and
  factory intrinsics, as the original script does.
- **Board distance.** Keep the board 0.4–1.0 m from the camera and vary the viewing angle (15–20 poses). At 2 m, one
  pixel is ~2.6 mm on the board.

## 한국어 요약
- 원본 캘리브레이션 폴더(`vision/zed_calibration`, `../ZED`)는 건드리지 않고, 옆에 만든 확인·재캡처용 도구입니다.
- **로봇은 읽기만 합니다.** RTDE로 자세만 읽고 로봇을 움직이지 않습니다. 팔은 티치 펜던트나 프리드라이브로
  직접 움직이세요.
- **일관성 검사.** 보드는 고정되어 있으므로, 팔을 여러 자세로 옮겨도 현재 캘리브레이션을 거쳐 계산한 보드
  위치가 같아야 합니다. 화면의 CONSISTENCY 값(mm/deg)이 작을수록 캘리브레이션이 정확합니다. 원래 세션은
  3.4 ± 2.1 mm였습니다.
- **키:**
  - `y`: 현재 자세 캡처
  - `h`: 모은 캡처로 새 핸드-아이 캘리브레이션 계산 (세션 폴더에만 저장)
  - `r`: 샘플 초기화
  - `q`: 종료
- **SDK가 없으면 숫자는 근사치입니다.** ZED SDK 권한이 없으면 렌즈 왜곡이 보정되지 않은 원본(raw) 영상을
  쓰므로 숫자가 근사치입니다. 최종 캘리브레이션은 SDK 권한을 받은 뒤 하세요.
- **보드 거리.** 보드는 카메라에서 0.4–1.0 m 거리가 좋습니다.
