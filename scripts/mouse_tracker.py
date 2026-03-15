#!/usr/bin/env python3
"""
Multi-camera mouse tracker for home footage review.

Shows all camera feeds in a security-camera grid, synchronised by wall-clock
time embedded in the SRT subtitle track of each MP4.

Each camera runs in a dedicated child process so seeks and H.264 decodes
happen in parallel; the main process only composites and displays frames.

Keyboard controls (mplayer-style):
  Space        pause / unpause
  ← / →        seek ±10 s
  ↑ / ↓        seek ±1 min
  PgUp / PgDn  seek ±10 min
  1–9          set speed 1×–9×
  0            set speed 10×
  a            auto mode (default)
  q / Esc      quit
"""
from __future__ import annotations

import argparse
import datetime
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Display ───────────────────────────────────────────────────────────────────
CELL_W = 640
CELL_H = 360
STATUS_H = 50

# ── Motion-detection reference window ────────────────────────────────────────
MOTION_REF_FRAMES = 5   # diff is always computed against a frame ≤ this many frames back

# ── Auto-mode frame-skip counts ───────────────────────────────────────────────
AUTO_NO_MOTION_FRAMES = 15   # display 1 every 15 frames when nothing is moving
AUTO_MOTION_FRAMES    = 5    # display 1 every 5 frames when motion is detected

# ── Motion detection parameters ───────────────────────────────────────────────
BLUR_KERNEL      = 5    # Gaussian blur kernel (must be odd)
DIFF_THRESHOLD   = 15   # per-pixel abs-diff threshold (0–255)
MIN_CONTOUR_AREA = 25   # minimum contour area in pixels

# ── Key codes returned by cv2.waitKeyEx on Linux/X11 ─────────────────────────
KEY_LEFT    = 65361
KEY_RIGHT   = 65363
KEY_UP      = 65362
KEY_DOWN    = 65364
KEY_PGUP    = 65365
KEY_PGDN    = 65366
KEY_ESC     = 27
KEY_SPACE   = 32


# ─────────────────────────────────────────────────────────────────────────────
# SRT parsing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SrtEntry:
    vid_start: float          # seconds within the video file
    vid_end: float
    wallclock: datetime.datetime  # wall-clock time at vid_start


def _srt_ts_to_seconds(s: str) -> float:
    """Parse HH:MM:SS,mmm or HH:MM:SS.mmm → float seconds."""
    m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)', s.strip())
    if not m:
        return 0.0
    h, mi, se, ms_raw = m.groups()
    ms = int(ms_raw.ljust(3, '0')[:3])
    return int(h) * 3600 + int(mi) * 60 + int(se) + ms / 1000.0


def _parse_srt(content: str) -> list[SrtEntry]:
    entries: list[SrtEntry] = []
    for block in re.split(r'\n{2,}', content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r'(.+?)\s*-->\s*(.+)', lines[1])
        if not m:
            continue
        vid_start = _srt_ts_to_seconds(m.group(1))
        vid_end   = _srt_ts_to_seconds(m.group(2))
        try:
            wc = datetime.datetime.strptime(lines[2], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
        entries.append(SrtEntry(vid_start, vid_end, wc))
    return entries


def _extract_srt(video_path: str) -> list[SrtEntry]:
    """Extract the SRT subtitle track from an MP4 via ffmpeg."""
    fd, srt_path = tempfile.mkstemp(suffix='.srt')
    os.close(fd)
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', video_path, '-map', '0:s:0', srt_path],
            capture_output=True,
        )
        if result.returncode != 0 or not os.path.exists(srt_path):
            return []
        with open(srt_path) as f:
            return _parse_srt(f.read())
    finally:
        if os.path.exists(srt_path):
            os.unlink(srt_path)


def _wallclock_to_vid_seconds(
    entries: list[SrtEntry], wc: datetime.datetime
) -> Optional[float]:
    for e in entries:
        seg_dur = e.vid_end - e.vid_start
        wc_end  = e.wallclock + datetime.timedelta(seconds=seg_dur)
        if e.wallclock <= wc < wc_end:
            return e.vid_start + (wc - e.wallclock).total_seconds()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Camera worker — runs entirely in a child process
# ─────────────────────────────────────────────────────────────────────────────

def _worker_loop(
    path: str,
    srt: list[SrtEntry],
    fps: float,
    total_frames: int,
    cell_w: int,
    cell_h: int,
    blur_kernel: int,
    diff_threshold: int,
    min_contour_area: int,
    motion_ref_frames: int,
    cmd_q: 'mp.Queue[Optional[float]]',
    result_q: 'mp.Queue[tuple]',
) -> None:
    """Runs in a child process. Owns the VideoCapture for one camera."""
    cap        = cv2.VideoCapture(path)
    prev_gray: Optional[np.ndarray] = None
    prev_vid_s: float = -1.0

    while True:
        cmd = cmd_q.get()
        if cmd is None:   # sentinel → clean shutdown
            break

        wc    = datetime.datetime.fromtimestamp(cmd)
        vid_s = _wallclock_to_vid_seconds(srt, wc)

        if vid_s is None:
            result_q.put((None, False, []))
            continue

        frame_num = max(0, min(int(vid_s * fps), total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            result_q.put((None, False, []))
            continue

        frame = cv2.resize(frame, (cell_w, cell_h))

        # Motion detection — always diff against a reference ≤ motion_ref_frames back
        # so accuracy is independent of how far we jumped for display.
        motion_ref_secs = motion_ref_frames / fps
        gap = vid_s - prev_vid_s
        if prev_gray is not None and (gap > motion_ref_secs or gap < 0):
            ref_vid_s = max(0.0, vid_s - motion_ref_secs)
            ref_num   = max(0, min(int(ref_vid_s * fps), total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, ref_num)
            ret_ref, ref_frame = cap.read()
            if ret_ref:
                ref_resized = cv2.resize(ref_frame, (cell_w, cell_h))
                prev_gray = cv2.GaussianBlur(
                    cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY),
                    (blur_kernel, blur_kernel), 0,
                )

        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (blur_kernel, blur_kernel), 0,
        )

        has_motion   = False
        motion_rects: list[tuple[int, int, int, int]] = []
        if prev_gray is not None:
            diff    = cv2.absdiff(prev_gray, gray)
            _, thr  = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thr, None, iterations=2)
            cnts, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            big          = [c for c in cnts if cv2.contourArea(c) > min_contour_area]
            has_motion   = bool(big)
            motion_rects = [cv2.boundingRect(c) for c in big]

        prev_gray  = gray
        prev_vid_s = vid_s

        result_q.put((frame, has_motion, motion_rects))

    cap.release()


class CameraWorker:
    """Main-process proxy for a camera running in a dedicated child process."""

    def __init__(
        self,
        path: str,
        cell_w: int,
        cell_h: int,
        blur_kernel: int,
        diff_threshold: int,
        min_contour_area: int,
    ) -> None:
        # Extract metadata in the main process before forking.
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {path}')
        self.fps: float        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.srt = _extract_srt(path)
        if not self.srt:
            raise RuntimeError(
                f'No subtitle track in {path}. '
                'File must be produced by retrieve_camera_footage.'
            )

        self.label        = Path(path).stem
        self.has_motion   = False
        self.motion_rects: list[tuple[int, int, int, int]] = []

        self._cmd_q:    mp.Queue = mp.Queue()
        self._result_q: mp.Queue = mp.Queue()

        self._proc = mp.Process(
            target=_worker_loop,
            args=(
                path, self.srt, self.fps, self.total_frames,
                cell_w, cell_h,
                blur_kernel, diff_threshold, min_contour_area,
                MOTION_REF_FRAMES,
                self._cmd_q, self._result_q,
            ),
            daemon=True,
        )
        self._proc.start()

    @property
    def wc_start(self) -> datetime.datetime:
        return self.srt[0].wallclock

    @property
    def wc_end(self) -> datetime.datetime:
        e = self.srt[-1]
        return e.wallclock + datetime.timedelta(seconds=e.vid_end - e.vid_start)

    def request(self, wc: datetime.datetime) -> None:
        """Send a frame request to the worker (non-blocking)."""
        self._cmd_q.put(wc.timestamp())

    def collect(self) -> Optional[np.ndarray]:
        """Block until the worker's result arrives. Updates has_motion / motion_rects."""
        frame, self.has_motion, self.motion_rects = self._result_q.get()
        return frame

    def stop(self) -> None:
        try:
            self._cmd_q.put(None)
            self._proc.join(timeout=2)
        finally:
            self._proc.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grid_cols(n: int) -> int:
    if n <= 1:
        return 1
    if n <= 2:
        return 2
    if n <= 4:
        return 2
    return 3


def _blank_cell(label: str) -> np.ndarray:
    cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    cv2.putText(cell, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(cell, 'N/A', (CELL_W // 2 - 35, CELL_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 40, 40), 2, cv2.LINE_AA)
    return cell


def _draw_cell(frame: np.ndarray, cam: CameraWorker) -> np.ndarray:
    out = frame.copy()
    for (x, y, w, h) in cam.motion_rects:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.putText(out, cam.label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, cam.label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if cam.has_motion:
        cv2.circle(out, (CELL_W - 18, 18), 10, (0, 0, 180), -1)
        cv2.circle(out, (CELL_W - 18, 18), 10, (0, 0, 255), 2)
    return out


def _make_grid(cells: list[np.ndarray], n_cols: int) -> np.ndarray:
    n_rows = math.ceil(len(cells) / n_cols)
    blank  = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    padded = cells + [blank] * (n_rows * n_cols - len(cells))
    rows   = [np.hstack(padded[r * n_cols:(r + 1) * n_cols]) for r in range(n_rows)]
    return np.vstack(rows)


def _make_status(
    wc: datetime.datetime,
    mode: str,
    any_motion: bool,
    total_w: int,
    paused: bool = False,
) -> np.ndarray:
    bar = np.zeros((STATUS_H, total_w, 3), dtype=np.uint8)

    cv2.putText(bar, wc.strftime('%Y-%m-%d  %H:%M:%S'), (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1, cv2.LINE_AA)

    if paused:
        badge_color = (0, 140, 200)
        badge_text  = '  PAUSED  '
    elif any_motion:
        badge_color = (30, 30, 180)
        badge_text  = '  MOTION  '
    else:
        badge_color = (30, 100, 30)
        badge_text  = '   quiet  '
    cv2.putText(bar, badge_text, (total_w // 2 - 60, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 1, cv2.LINE_AA)

    if not paused:
        cv2.putText(bar, mode, (total_w - 280, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 200, 140), 1, cv2.LINE_AA)

    return bar


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global CELL_W, CELL_H, BLUR_KERNEL, DIFF_THRESHOLD, MIN_CONTOUR_AREA  # noqa: PLW0603
    ap = argparse.ArgumentParser(
        description='Multi-camera mouse tracker — review home footage efficiently.'
    )
    ap.add_argument('videos', nargs='+', help='MP4 files, one per camera')
    ap.add_argument(
        '--auto-no-motion-frames', type=int, default=AUTO_NO_MOTION_FRAMES,
        metavar='N',
        help=f'Frames to skip per tick when no motion in auto mode (default {AUTO_NO_MOTION_FRAMES})',
    )
    ap.add_argument(
        '--auto-motion-frames', type=int, default=AUTO_MOTION_FRAMES,
        metavar='N',
        help=f'Frames to skip per tick when motion detected in auto mode (default {AUTO_MOTION_FRAMES})',
    )
    ap.add_argument(
        '--start', metavar='ISO',
        help='Starting wall-clock time, e.g. 2024-02-10T10:30:00 (default: earliest)',
    )
    ap.add_argument('--cell-width',  type=int, default=640)
    ap.add_argument('--cell-height', type=int, default=360)
    ap.add_argument(
        '--blur', type=int, default=5, metavar='N',
        help='Gaussian blur kernel size for motion detection (odd, default 5)',
    )
    ap.add_argument(
        '--diff-threshold', type=int, default=15, metavar='N',
        help='Per-pixel diff threshold 0–255 (default 15)',
    )
    ap.add_argument(
        '--min-area', type=int, default=25, metavar='N',
        help='Minimum contour area in pixels to count as motion (default 25)',
    )
    args = ap.parse_args()

    CELL_W           = args.cell_width
    CELL_H           = args.cell_height
    BLUR_KERNEL      = args.blur if args.blur % 2 == 1 else args.blur + 1
    DIFF_THRESHOLD   = args.diff_threshold
    MIN_CONTOUR_AREA = args.min_area

    # ── Spawn camera workers ───────────────────────────────────────────────────
    cameras: list[CameraWorker] = []
    for path in args.videos:
        print(f'Loading {path} …')
        try:
            cam = CameraWorker(
                path,
                cell_w=CELL_W, cell_h=CELL_H,
                blur_kernel=BLUR_KERNEL,
                diff_threshold=DIFF_THRESHOLD,
                min_contour_area=MIN_CONTOUR_AREA,
            )
        except RuntimeError as exc:
            sys.exit(str(exc))
        cameras.append(cam)
        print(f'  {cam.wc_start:%Y-%m-%d %H:%M:%S}  →  {cam.wc_end:%Y-%m-%d %H:%M:%S}')

    if not cameras:
        sys.exit('No cameras loaded.')

    # ── Timeline ──────────────────────────────────────────────────────────────
    global_start = min(c.wc_start for c in cameras)
    global_end   = max(c.wc_end   for c in cameras)

    if args.start:
        current_wc = datetime.datetime.fromisoformat(args.start)
        current_wc = max(global_start, min(global_end, current_wc))
    else:
        current_wc = global_start

    fps = cameras[0].fps

    print(f'\nTimeline: {global_start:%Y-%m-%d %H:%M:%S}  →  {global_end:%Y-%m-%d %H:%M:%S}')
    print(f'FPS: {fps:.2f}  |  auto quiet: 1/{args.auto_no_motion_frames} frames'
          f'  |  auto motion: 1/{args.auto_motion_frames} frames')
    print(f'Worker processes: {len(cameras)}')
    print('\nControls (mplayer-style):')
    print('  Space        pause / unpause')
    print('  ← / →        seek ±10 s')
    print('  ↑ / ↓        seek ±1 min')
    print('  PgUp / PgDn  seek ±10 min')
    print('  1–9          set speed 1×–9×')
    print('  0            set speed 10×')
    print('  a            auto mode (default)')
    print('  q / Esc      quit\n')

    n_cols = _grid_cols(len(cameras))
    mode: str | int = 'auto'
    paused      = False
    force_fetch = True
    any_motion  = False
    cells: list[np.ndarray] = []
    actual_fps  = 0.0
    _last_tick  = time.monotonic()

    cv2.namedWindow('Mouse Tracker', cv2.WINDOW_NORMAL)

    try:
        while True:
            # ── Fetch frames from all workers in parallel ──────────────────────
            # Send requests to every worker simultaneously, then collect.
            # Workers decode in parallel; total latency = slowest camera, not sum.
            if not paused or force_fetch:
                force_fetch = False
                for cam in cameras:
                    cam.request(current_wc)
                any_motion = False
                cells = []
                for cam in cameras:
                    raw = cam.collect()
                    if raw is None:
                        cells.append(_blank_cell(cam.label))
                    else:
                        if cam.has_motion:
                            any_motion = True
                        cells.append(_draw_cell(raw, cam))

            # ── Compute step and waitKey delay ────────────────────────────────
            if mode == 'auto':
                frame_skip = args.auto_motion_frames if any_motion else args.auto_no_motion_frames
                step       = frame_skip / fps
                wait_ms    = 1
            else:
                frame_skip = 1
                step       = 1.0 / fps
                wait_ms    = max(1, int(1000 / (fps * mode)))

            # ── Render ────────────────────────────────────────────────────────
            if mode == 'auto':
                mode_label = f'auto  skip:{frame_skip}  ({frame_skip * fps:.0f} / {actual_fps:.0f} fps)'
            else:
                mode_label = f'{mode}×  ({fps * mode:.0f} / {actual_fps:.0f} fps)'
            grid   = _make_grid(cells, n_cols)
            status = _make_status(current_wc, mode_label, any_motion, grid.shape[1], paused)
            cv2.imshow('Mouse Tracker', np.vstack([grid, status]))

            # ── Measure actual display FPS (EMA) ──────────────────────────────
            now        = time.monotonic()
            dt         = now - _last_tick
            _last_tick = now
            if dt > 0:
                actual_fps = 0.1 * (1.0 / dt) + 0.9 * actual_fps

            # ── Handle input ──────────────────────────────────────────────────
            seek = None
            key  = cv2.waitKeyEx(30 if paused else wait_ms)
            if key in (ord('q'), KEY_ESC):
                break
            elif key == KEY_SPACE:
                paused = not paused
            elif key == ord('a'):
                mode = 'auto'
            elif ord('1') <= key <= ord('9'):
                mode = key - ord('0')
            elif key == ord('0'):
                mode = 10
            elif key == KEY_RIGHT:
                seek = 10.0
            elif key == KEY_LEFT:
                seek = -10.0
            elif key == KEY_UP:
                seek = 60.0
            elif key == KEY_DOWN:
                seek = -60.0
            elif key == KEY_PGUP:
                seek = 600.0
            elif key == KEY_PGDN:
                seek = -600.0

            # ── Advance timeline ──────────────────────────────────────────────
            if seek is not None:
                current_wc += datetime.timedelta(seconds=seek)
                force_fetch = True
            elif not paused:
                current_wc += datetime.timedelta(seconds=step)

            if current_wc < global_start:
                current_wc = global_start
            if current_wc >= global_end:
                print('Reached end of footage.')
                break

    finally:
        cv2.destroyAllWindows()
        for cam in cameras:
            cam.stop()


if __name__ == '__main__':
    main()
