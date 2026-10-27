"""Recording, the three-panel video and the live viewer HUD for run_librarian.py.

Video frame (1920 x 1080):
    | what the skin feels (560) | MuJoCo: main + close-up (800) | wrist camera + urx console (560) |
"""
from __future__ import annotations

import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from shear_agent.robot import mat_to_pose  # noqa: E402
from shear_agent.robot.render import SceneRenderer, arrow  # noqa: E402

SIDES = ("left", "right")
COLORS = {"left": "#1f77b4", "right": "#d62728"}
W, H = 1920, 1080


# ------------------------------------------------------------------ data
@dataclass
class Log:
    """Time series sampled by Recorder (one row per sample)."""
    rows: List[dict] = field(default_factory=list)

    def column(self, key):
        return np.array([r[key] for r in self.rows])

    def save_csv(self, path):
        if not self.rows:
            return
        keys = [k for k, v in self.rows[0].items() if np.ndim(v) == 0]
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in self.rows:
                f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")


class Recorder:
    """Samples robot, gripper, skin and object state at `rate_hz` of simulated time."""

    def __init__(self, rob, gripper, skins, book: str = "book_target", others=(), rate_hz: float = 100.0):
        self.rob, self.gripper, self.skins = rob, gripper, skins
        self.m, self.d = rob.m, rob.d
        self.book = self.m.body(book).id
        self.others = [self.m.body(n).id for n in others]
        self.others0 = {b: self.d.xpos[b].copy() for b in self.others}
        self.period = 1.0 / rate_hz
        self.t_next = 0.0
        self.log = Log()
        self.frames: Dict[str, object] = {}
        self.phase = "start"
        self.grasp_ref = None  # book position in the TCP frame when the grasp closes
        rob.callbacks.append(self._on_tick)

    def _book_in_tcp(self):
        T = self.rob.get_pose()
        return T[:3, :3].T @ (self.d.xpos[self.book] - T[:3, 3])

    def mark_grasp(self):
        self.grasp_ref = self._book_in_tcp()

    def release(self):
        self.grasp_ref = None

    def _on_tick(self, rob):
        if rob.time + 1e-9 < self.t_next:
            return
        self.t_next = rob.time + self.period
        tact = {s: k.read(self.d) for s, k in self.skins.items()}
        self.frames = tact
        f = np.asarray(rob.get_tcp_force())
        slip = np.linalg.norm(self._book_in_tcp() - self.grasp_ref) if self.grasp_ref is not None else 0.0
        disturb = max((np.linalg.norm(self.d.xpos[b] - p0) for b, p0 in self.others0.items()), default=0.0)
        row = dict(t=rob.time, phase=self.phase, gripper_pos=self.gripper.get_position(),
                   gripper_req=self.gripper.get_requested_position(), gripper_force=int(self.gripper.force),
                   gripper_obj=self.gripper.get_object_status(),
                   fx=f[0], fy=f[1], fz=f[2], tx=f[3], ty=f[4], tz=f[5],
                   slip_mm=1e3 * slip, neighbours_mm=1e3 * disturb,
                   book_x=self.d.xpos[self.book][0], book_y=self.d.xpos[self.book][1], book_z=self.d.xpos[self.book][2])
        for i, q in enumerate(rob.getj()):
            row[f"q{i + 1}"] = q
        for side, fr in tact.items():
            row[f"{side}_normal"] = fr.total_normal
            row[f"{side}_shear"] = float(np.linalg.norm(fr.total_shear))
            row[f"{side}_shear_x"], row[f"{side}_shear_y"] = (float(v) for v in fr.total_shear)
            row[f"{side}_margin"] = fr.net_margin()
            row[f"{side}_margin_local"] = fr.grasp_margin()
            row[f"{side}_pen_um"] = 1e6 * float(fr.penetration.max())
            raw = getattr(self.skins[side], "raw", None)
            if raw is not None:  # the unfiltered sample (every control tick) for diagnostics
                row[f"{side}_normal_raw"] = raw.total_normal
                row[f"{side}_shear_raw"] = float(np.linalg.norm(raw.total_shear))
        self.log.rows.append(row)


# ------------------------------------------------------------------ panels
class SensorPanel:
    """'What the skin feels': taxel maps + time series (matplotlib, redrawn per frame)."""

    def __init__(self, width: int = 560, height: int = 1026, window: float = 12.0, q_star: float = 0.5,
                 taxel_max: float = 4.0, arrow_mm_per_n: float = 6.0):
        self.window, self.arrow_gain = window, arrow_mm_per_n
        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#101418")
        gs = self.fig.add_gridspec(5, 2, height_ratios=[2.25, 1, 1, 1, 1], hspace=0.55, wspace=0.25,
                                   left=0.13, right=0.97, top=0.93, bottom=0.05)
        txt = dict(color="#e8e8e8")
        self.fig.suptitle("What the skin feels (PDMS 5 x 7 taxels, 20 Hz frames)", fontsize=12, **txt)
        self.maps, self.quivers, self.labels = {}, {}, {}
        for k, side in enumerate(SIDES):
            ax = self.fig.add_subplot(gs[0, k])
            self._style(ax)
            im = ax.imshow(np.zeros((7, 5)), cmap="magma", vmin=0, vmax=taxel_max, origin="lower",
                           extent=(-25.5, 25.5, -35.5, 35.5), aspect="equal")
            xs = np.linspace(-20.4, 20.4, 5)
            ys = np.linspace(-30.4, 30.4, 7)
            X, Y = np.meshgrid(xs, ys)
            self.quivers[side] = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), color="#4fd1ff",
                                           angles="xy", scale_units="xy", scale=1.0, width=0.018)
            ax.set_title(f"{side} blade", fontsize=9, color=COLORS[side] if side == "left" else "#ff8080", pad=4)
            ax.set_xlabel("across (mm)", fontsize=8, **txt)
            if k == 0:
                ax.set_ylabel("along finger (mm), tip up", fontsize=8, **txt)
            self.maps[side] = im
            self.labels[side] = ax.text(0.5, -0.32, "", transform=ax.transAxes, fontsize=7.5, color="#e8e8e8",
                                        ha="center", va="top")
        cb = self.fig.colorbar(self.maps["right"], ax=[self.fig.axes[0], self.fig.axes[1]], fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors="#e8e8e8")
        cb.set_label(f"normal per taxel (N); arrows: shear, {arrow_mm_per_n:g} mm/N", fontsize=7.5, **txt)

        self.ax_n = self.fig.add_subplot(gs[1, :])
        self.ax_s = self.fig.add_subplot(gs[2, :])
        self.ax_q = self.fig.add_subplot(gs[3, :])
        self.ax_w = self.fig.add_subplot(gs[4, :])
        self.lines = {}
        for ax, title, ylab in [(self.ax_n, "normal force on each skin", "N"),
                                (self.ax_s, "shear (friction) force on each skin", "N"),
                                (self.ax_q, "margin q = 1 - |shear| / (mu normal) per blade  &  grip force", "q"),
                                (self.ax_w, "arm F/T (book load) and book slip", "N / mm")]:
            self._style(ax)
            ax.set_title(title, fontsize=9, color="#e8e8e8", loc="left")
            ax.set_ylabel(ylab, fontsize=8, **txt)
        for side in SIDES:
            self.lines[f"n_{side}"], = self.ax_n.plot([], [], color=COLORS[side], lw=1.6, label=side)
            self.lines[f"s_{side}"], = self.ax_s.plot([], [], color=COLORS[side], lw=1.6, label=side)
            self.lines[f"q_{side}"], = self.ax_q.plot([], [], color=COLORS[side], lw=1.6, label=f"Q {side}")
        self.ax_q.axhline(q_star, color="#bbbbbb", lw=0.8, ls="--")
        self.ax_q.text(0.99, q_star, "q*", transform=self.ax_q.get_yaxis_transform(), color="#bbbbbb",
                       fontsize=7, ha="right", va="bottom")
        self.ax_f = self.ax_q.twinx()
        self.lines["force_reg"], = self.ax_f.plot([], [], color="#f2c94c", lw=1.2, ls="-", label="force register")
        self.ax_f.set_ylim(-5, 260)
        self.ax_f.tick_params(labelsize=7, colors="#f2c94c")
        self.lines["fz"], = self.ax_w.plot([], [], color="#9be7a1", lw=1.4, label="-Fz (N)")
        self.lines["fy"], = self.ax_w.plot([], [], color="#c39bd3", lw=1.2, label="-Fy pull (N)")
        self.lines["slip"], = self.ax_w.plot([], [], color="#ffb86c", lw=1.4, label="slip (mm)")
        for ax in (self.ax_n, self.ax_s, self.ax_w):
            ax.legend(fontsize=7, loc="upper left", facecolor="#101418", labelcolor="#e8e8e8", framealpha=0.6, ncol=3)
        self.ax_q.legend(handles=[self.lines["q_left"], self.lines["q_right"], self.lines["force_reg"]], fontsize=7,
                         loc="lower left", facecolor="#101418", labelcolor="#e8e8e8", framealpha=0.6, ncol=3)
        self.ax_q.set_ylim(-0.1, 1.05)

    @staticmethod
    def _style(ax):
        ax.set_facecolor("#171c22")
        ax.tick_params(labelsize=7, colors="#cfcfcf")
        for sp in ax.spines.values():
            sp.set_color("#555")
        ax.grid(alpha=0.18)

    def draw(self, frames, log: Log) -> np.ndarray:
        for side, fr in frames.items():
            self.maps[side].set_data(fr.normal)
            uv = fr.shear * self.arrow_gain                                  # mm on the map
            n = np.linalg.norm(uv, axis=-1, keepdims=True)
            uv = uv * np.minimum(1.0, 14.0 / np.maximum(n, 1e-9))            # clip at 14 mm (taxel pitch ~10 mm)
            self.quivers[side].set_UVC(uv[..., 0], uv[..., 1])
            self.labels[side].set_text(f"N {fr.total_normal:5.1f} N  |T| {np.linalg.norm(fr.total_shear):4.2f} N  "
                                       f"q {fr.net_margin():+.2f}")
        rows = log.rows
        if rows:
            t = log.column("t")
            t1 = t[-1]
            t0 = max(0.0, t1 - self.window)
            sel = t >= t0
            ts = t[sel]
            for side in SIDES:
                self.lines[f"n_{side}"].set_data(ts, log.column(f"{side}_normal")[sel])
                self.lines[f"s_{side}"].set_data(ts, log.column(f"{side}_shear")[sel])
                q = np.clip(log.column(f"{side}_margin")[sel], -0.1, 1.05)
                self.lines[f"q_{side}"].set_data(ts, q)
            self.lines["force_reg"].set_data(ts, log.column("gripper_force")[sel])
            self.lines["fz"].set_data(ts, -log.column("fz")[sel])
            self.lines["fy"].set_data(ts, -log.column("fy")[sel])
            self.lines["slip"].set_data(ts, log.column("slip_mm")[sel])
            for ax in (self.ax_n, self.ax_s, self.ax_q, self.ax_w):
                ax.set_xlim(t0, max(t0 + self.window, t1))
            nmax = max(5.0, np.max([log.column(f"{s}_normal")[sel].max() for s in SIDES]) * 1.15)
            smax = max(1.0, np.max([log.column(f"{s}_shear")[sel].max() for s in SIDES]) * 1.25)
            wmax = max(2.0, np.max(np.abs(np.r_[log.column("fz")[sel], log.column("fy")[sel],
                                                log.column("slip_mm")[sel]])) * 1.2)
            self.ax_n.set_ylim(0, nmax)
            self.ax_s.set_ylim(0, smax)
            self.ax_w.set_ylim(-wmax * 0.3, wmax)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()


def _font(size: int, bold: bool = False):
    for name in (["DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"] if bold else ["DejaVuSansMono.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.truetype(str(matplotlib.get_data_path()) + "/fonts/ttf/DejaVuSansMono.ttf", size)
    except OSError:
        return ImageFont.load_default()


class Compositor:
    """Lays out the three panels, the title bar and the urx console."""

    def __init__(self, renderer: SceneRenderer, main_cam, close_cam, wrist_cam: str = "wrist_cam",
                 title: str = ""):
        self.r = renderer
        self.main_cam, self.close_cam, self.wrist_cam = main_cam, close_cam, wrist_cam
        self.title = title
        self.panel = SensorPanel()
        self.f_title, self.f_head = _font(22, True), _font(17, True)
        self.f_mono, self.f_small = _font(15), _font(13)

    def force_arrows(self, rec: Recorder, scale: float = 0.0012, shear_gain: float = 3.0):
        """Per blade: squeeze force on the book (blue) and friction on the book x3 (orange), at the skin's CoP."""
        out = []
        d = rec.d
        for side, fr in rec.frames.items():
            if not fr.in_contact:
                continue
            body = getattr(rec.skins[side], "skin", rec.skins[side]).body
            R, p = d.xmat[body].reshape(3, 3), d.xpos[body]
            cop = p + R @ np.r_[fr.cop, -0.00025]
            n_on_book = R @ np.array([0.0, 0.0, -fr.total_normal])     # skin pushes the book along -z_skin
            t_on_book = R @ np.r_[-fr.total_shear, 0.0]                # friction on the book opposes the drag
            out.append(arrow(cop - n_on_book * scale, n_on_book * scale, (0.25, 0.6, 1.0, 1.0), 0.005))
            if np.linalg.norm(fr.total_shear) > 0.05:
                out.append(arrow(cop, t_on_book * scale * shear_gain, (1.0, 0.55, 0.1, 1.0), 0.004))
        f = np.asarray(rec.rob.get_tcp_force()[:3])
        if np.linalg.norm(f) > 0.5:
            tcp = rec.rob.get_pose()[:3, 3]
            out.append(arrow(tcp + np.array([0, 0, 0.09]), f * 0.01, (0.55, 0.95, 0.6, 1.0), 0.004))
        return out

    def frame(self, rec: Recorder) -> np.ndarray:
        rob, d = rec.rob, rec.d
        canvas = Image.new("RGB", (W, H), (16, 20, 24))
        draw = ImageDraw.Draw(canvas)
        # title bar
        draw.text((18, 14), self.title, font=self.f_title, fill=(235, 235, 235))
        draw.text((W - 430, 16), f"t = {rob.time:6.2f} s   phase: {rec.phase}", font=self.f_head, fill=(242, 201, 76))
        top = 54
        # left: sensor panel
        canvas.paste(Image.fromarray(self.panel.draw(rec.frames, rec.log)), (0, top))
        # centre: MuJoCo views
        extra = self.force_arrows(rec)
        main = self.r.render(d, self.main_cam(rob.time) if callable(self.main_cam) else self.main_cam, (800, 600),
                             extra_geoms=extra)
        close = self.r.render(d, self.close_cam(rob.time) if callable(self.close_cam) else self.close_cam,
                              (800, 420), extra_geoms=extra)
        canvas.paste(Image.fromarray(main), (560, top))
        canvas.paste(Image.fromarray(close), (560, top + 606))
        draw.text((572, top + 8), "MuJoCo (Filament PBR)" if self.r.backend == "filament" else "MuJoCo",
                  font=self.f_head, fill=(255, 255, 255))
        draw.text((572, top + 614), "close-up (follows the tool)", font=self.f_head, fill=(255, 255, 255))
        draw.text((572, top + 580), "blue: squeeze on book (12 mm/10 N)   orange: friction on book (x3)   green: arm F/T",
                  font=self.f_small, fill=(220, 220, 220))
        # right: wrist camera + console
        wrist = self.r.render(d, self.wrist_cam, (560, 420))
        canvas.paste(Image.fromarray(wrist), (1360, top))
        draw.text((1372, top + 8), "What the wrist camera sees", font=self.f_head, fill=(255, 255, 255))
        self._console(draw, rec, x=1372, y=top + 432)
        return np.asarray(canvas)

    def _console(self, draw, rec: Recorder, x: int, y: int):
        rob, g = rec.rob, rec.gripper
        q = np.degrees(rob.getj())
        pose = rob.getl()
        f = rob.get_tcp_force()
        lines = [("urx console (same calls run the real UR10e)", (242, 201, 76), self.f_head)]
        lines.append(("rob.getj()  [deg]", (170, 170, 170), self.f_small))
        lines.append(("  " + "  ".join(f"{v:7.2f}" for v in q[:3]), (230, 230, 230), self.f_mono))
        lines.append(("  " + "  ".join(f"{v:7.2f}" for v in q[3:]), (230, 230, 230), self.f_mono))
        lines.append(("rob.getl()  [m, rotvec]", (170, 170, 170), self.f_small))
        lines.append(("  " + " ".join(f"{v:6.3f}" for v in pose), (230, 230, 230), self.f_mono))
        lines.append(("rob.get_tcp_force()  [N, N m]", (170, 170, 170), self.f_small))
        lines.append(("  " + " ".join(f"{v:6.1f}" for v in f), (230, 230, 230), self.f_mono))
        obj = {0: "moving", 1: "object (opening)", 2: "object (closing)", 3: "at position"}[g.get_object_status()]
        lines.append((f"gripper  pos {g.get_position():3d}/255  req {g.get_requested_position():3d}  "
                      f"force {int(g.force):3d}", (230, 230, 230), self.f_small))
        lines.append((f"         {obj}", (230, 230, 230), self.f_small))
        lines.append(("commands", (170, 170, 170), self.f_small))
        for t, text in rob.command_log[-9:]:
            short = text if len(text) < 62 else text[:59] + "..."
            lines.append((f"{t:6.2f} {short}", (155, 220, 255), self.f_small))
        for text, color, font in lines:
            draw.text((x, y), text, font=font, fill=color)
            y += font.size + 6


class VideoWriter:
    """Raw RGB frames -> H.264 MP4 through ffmpeg (system ffmpeg or the imageio-ffmpeg binary)."""

    def __init__(self, path, fps: int = 30, crf: int = 20):
        exe = shutil.which("ffmpeg")
        if exe is None:
            import imageio_ffmpeg  # pip install imageio-ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        self.p = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset",
                                   "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                   str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(frame[:H, :W, :3]).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


# ------------------------------------------------------------------ live viewer HUD
class ViewerHUD:
    """Interactive MuJoCo viewer (local machine): real-time pacing, text overlay with the urx state,
    live force plots and the skin's taxel maps. macOS: run the script with `mjpython`."""

    def __init__(self, rec: Recorder, realtime: float = 1.0, fps: float = 30.0):
        import time

        import mujoco.viewer

        self.rec, self.realtime, self.period = rec, realtime, 1.0 / fps
        self.viewer = mujoco.viewer.launch_passive(rec.m, rec.d, show_left_ui=False, show_right_ui=False)
        with self.viewer.lock():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.viewer.cam.lookat[:] = (0.0, 0.75, 0.45)
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 1.6, 200, -20
        self.fig_n, self.fig_q = mujoco.MjvFigure(), mujoco.MjvFigure()
        for fig, title in ((self.fig_n, "skin normal / shear (N)"), (self.fig_q, "grasp margin Q")):
            fig.title = title
            fig.flg_extend = 1
            fig.figurergba[3] = 0.5
            fig.gridsize[:] = [5, 5]
        self._clock = time.perf_counter
        self.t_wall0, self.t_sim0 = self._clock(), rec.rob.time
        self.t_next = 0.0
        rec.rob.callbacks.append(self._on_tick)

    def _on_tick(self, rob):
        if rob.time < self.t_next:
            return
        self.t_next = rob.time + self.period
        if not self.viewer.is_running():
            raise KeyboardInterrupt("viewer closed")
        lag = (rob.time - self.t_sim0) / self.realtime - (self._clock() - self.t_wall0)
        if lag > 0:
            import time
            time.sleep(lag)
        log = self.rec.log.rows[-300:]
        for k, fig in enumerate((self.fig_n, self.fig_q)):
            keys = (["left_normal", "right_normal", "left_shear", "right_shear"] if k == 0
                    else ["left_margin", "right_margin"])
            for i, key in enumerate(keys):
                n = len(log)
                fig.linepnt[i] = n
                fig.linename[i] = key
                for j, r in enumerate(log):
                    fig.linedata[i][2 * j] = r["t"]
                    fig.linedata[i][2 * j + 1] = max(-0.2, r[key])
        q = np.degrees(rob.getj())
        f = rob.get_tcp_force()
        g = self.rec.gripper
        left = "\n".join(["phase", "getj() deg", "", "getl()", "TCP force", "gripper", "last command"])
        right = "\n".join([self.rec.phase, " ".join(f"{v:6.1f}" for v in q[:3]), " ".join(f"{v:6.1f}" for v in q[3:]),
                           " ".join(f"{v:.3f}" for v in rob.getl()), " ".join(f"{v:.1f}" for v in f[:3]) + " N",
                           f"pos {g.get_position()} req {g.get_requested_position()} force {int(g.force)}",
                           rob.command_log[-1][1][:48] if rob.command_log else ""])
        maps = [heatmap_image(fr.normal, 6.0) for fr in self.rec.frames.values()]
        with self.viewer.lock():
            self.viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right)])
            self.viewer.set_figures([(mujoco.MjrRect(0, 0, 420, 260), self.fig_n),
                                     (mujoco.MjrRect(0, 260, 420, 220), self.fig_q)])
            if maps:
                imgs = []
                for k, img in enumerate(maps):
                    imgs.append((mujoco.MjrRect(430 + k * 130, 0, img.shape[1], img.shape[0]), img))
                self.viewer.set_images(imgs)
        self.viewer.sync()

    def close(self):
        self.viewer.close()


def heatmap_image(normal: np.ndarray, vmax: float, cell: int = 16) -> np.ndarray:
    """Taxel map -> small RGB image (no GL needed), finger tip at the top."""
    cmap = plt.get_cmap("magma")
    rgb = (cmap(np.clip(normal / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
    return np.ascontiguousarray(rgb[::-1])
