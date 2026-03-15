#!/usr/bin/env python3
"""
Mice detector — plays a video file and runs YOLO inference to detect mice.

Processes 1 every --frame-skip frames (default 10) and overlays bounding boxes
on the display for all frames using the last known detections.

Press q or Esc to quit.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

STATUS_H = 40
KEY_ESC = 27


# ── SRT parsing (wall-clock time from subtitle track) ─────────────────────────

def _srt_ts_to_seconds(s: str) -> float:
    m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)', s.strip())
    if not m:
        return 0.0
    h, mi, se, ms_raw = m.groups()
    ms = int(ms_raw.ljust(3, '0')[:3])
    return int(h) * 3600 + int(mi) * 60 + int(se) + ms / 1000.0


def _parse_srt(content: str) -> list[tuple[float, float, datetime.datetime]]:
    """Return list of (vid_start_s, vid_end_s, wallclock) tuples."""
    entries = []
    for block in re.split(r'\n{2,}', content.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r'(.+?)\s*-->\s*(.+)', lines[1])
        if not m:
            continue
        vid_start = _srt_ts_to_seconds(m.group(1))
        vid_end = _srt_ts_to_seconds(m.group(2))
        try:
            wc = datetime.datetime.strptime(lines[2], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
        entries.append((vid_start, vid_end, wc))
    return entries


def _extract_srt(video_path: str) -> list[tuple[float, float, datetime.datetime]]:
    """Extract SRT from video, caching the result alongside the video as <video>.srt."""
    srt_path = video_path + '.srt'
    if not os.path.exists(srt_path):
        print(f'Extracting subtitle track → {srt_path} …')
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', video_path, '-map', '0:s:0', srt_path],
            capture_output=True,
        )
        if result.returncode != 0 or not os.path.exists(srt_path):
            print('Warning: no subtitle track found, wall-clock time unavailable.')
            return []
    with open(srt_path) as f:
        return _parse_srt(f.read())


def _vid_seconds_to_wallclock(
    srt: list[tuple[float, float, datetime.datetime]], vid_s: float
) -> Optional[datetime.datetime]:
    for (vs, ve, wc) in srt:
        if vs <= vid_s < ve:
            return wc + datetime.timedelta(seconds=vid_s - vs)
    return None


def _wallclock_to_frame(
    srt: list[tuple[float, float, datetime.datetime]], wc: datetime.datetime, fps: float
) -> Optional[int]:
    for (vs, ve, entry_wc) in srt:
        seg_dur = ve - vs
        entry_end = entry_wc + datetime.timedelta(seconds=seg_dur)
        if entry_wc <= wc < entry_end:
            vid_s = vs + (wc - entry_wc).total_seconds()
            return int(vid_s * fps)
    return None


# ─────────────────────────────────────────────────────────────────────────────

class Detection(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    conf: float


def _run_inference(model: YOLO, frame: np.ndarray, conf: float) -> list[Detection]:
    results = model.predict(frame, conf=conf, verbose=False)
    detections: list[Detection] = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        names = r.names
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            label = names[cls_id] if names else str(cls_id)
            confidence = float(box.conf[0].item())
            detections.append(Detection(x1, y1, x2, y2, label, confidence))
    return detections


def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), (0, 200, 0), 2)
        text = f'{d.label} {d.conf:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(d.y1 - 6, th + 2)
        cv2.rectangle(out, (d.x1, ty - th - 2), (d.x1 + tw + 4, ty + 2), (0, 200, 0), -1)
        cv2.putText(out, text, (d.x1 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    if detections:
        cv2.circle(out, (out.shape[1] - 18, 18), 10, (0, 0, 180), -1)
        cv2.circle(out, (out.shape[1] - 18, 18), 10, (0, 0, 255), 2)
    return out


def _make_status(
    frame_idx: int,
    total_frames: int,
    fps: float,
    n_detections: int,
    model_name: str,
    width: int,
    wallclock: Optional[datetime.datetime] = None,
) -> np.ndarray:
    bar = np.zeros((STATUS_H, width, 3), dtype=np.uint8)
    time_s = frame_idx / fps if fps > 0 else 0
    h = int(time_s // 3600)
    m = int((time_s % 3600) // 60)
    s = int(time_s % 60)
    wc_str = wallclock.strftime('%Y-%m-%d %H:%M:%S') if wallclock else f'{h:02d}:{m:02d}:{s:02d}'
    progress = f'{wc_str}  frame {frame_idx}/{total_frames}'
    cv2.putText(bar, progress, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
    det_text = f'{n_detections} detection(s)' if n_detections else 'no detections'
    det_color = (0, 200, 0) if n_detections else (100, 100, 100)
    cv2.putText(bar, det_text, (width // 2 - 70, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 1, cv2.LINE_AA)
    cv2.putText(bar, model_name, (width - 220, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    return bar


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Detect mice in a video file using YOLO and display results.'
    )
    ap.add_argument('video', help='Path to the input video file')
    ap.add_argument(
        '--frame-skip', type=int, default=10, metavar='N',
        help='Run YOLO inference on 1 in every N frames (default: 10)',
    )
    ap.add_argument(
        '--model', default='yolo26n.pt', metavar='PATH',
        help='YOLO model weights file (default: yolo26n.pt)',
    )
    ap.add_argument(
        '--conf', type=float, default=0.25, metavar='FLOAT',
        help='Detection confidence threshold (default: 0.25)',
    )
    ap.add_argument(
        '--display-width', type=int, default=1280, metavar='W',
        help='Display window width in pixels (default: 1280)',
    )
    ap.add_argument(
        '--start', metavar='TIMESTAMP',
        help='Wall-clock time to start from, e.g. "2024-02-10 10:30:00"',
    )
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f'Video not found: {args.video}')
    if not os.path.exists(args.model):
        sys.exit(f'Model not found: {args.model}')

    print(f'Extracting/loading subtitle track from {args.video} …')
    srt = _extract_srt(args.video)

    print(f'Loading model {args.model} …')
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f'Cannot open video: {args.video}')

    src_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_h = int(args.display_width * src_h / src_w) if src_w else args.display_width * 9 // 16

    print(f'Video: {args.video}  ({src_w}×{src_h} @ {src_fps:.1f} fps, {total_frames} frames)')
    print(f'Frame skip: {args.frame_skip}  |  confidence: {args.conf}')
    print('Press q or Esc to quit.\n')

    model_name = Path(args.model).name
    last_detections: list[Detection] = []
    frame_idx = 0

    if args.start:
        if not srt:
            sys.exit('--start requires a subtitle track, but none was found.')
        try:
            start_wc = datetime.datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            sys.exit(f'--start must be in "YYYY-MM-DD HH:MM:SS" format, got: {args.start!r}')
        start_frame = _wallclock_to_frame(srt, start_wc, src_fps)
        if start_frame is None:
            sys.exit(f'Timestamp {args.start!r} is outside the subtitle track range.')
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        print(f'Seeking to {args.start} (frame {start_frame}) …')
    #wait_ms = max(1, int(1000 / src_fps))
    wait_ms = 1

    cv2.namedWindow('Mice Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mice Detector', args.display_width, disp_h + STATUS_H)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % args.frame_skip != 0:
                continue

            last_detections = _run_inference(model, frame, args.conf)

            display = cv2.resize(frame, (args.display_width, disp_h))
            # Scale detection coordinates to display size
            sx = args.display_width / src_w if src_w else 1.0
            sy = disp_h / src_h if src_h else 1.0
            scaled = [
                Detection(
                    int(d.x1 * sx), int(d.y1 * sy),
                    int(d.x2 * sx), int(d.y2 * sy),
                    d.label, d.conf,
                )
                for d in last_detections
            ]
            annotated = _draw_detections(display, scaled)
            vid_s = frame_idx / src_fps
            wallclock = _vid_seconds_to_wallclock(srt, vid_s)
            status = _make_status(
                frame_idx, total_frames, src_fps,
                len(last_detections), model_name, args.display_width,
                wallclock=wallclock,
            )
            cv2.imshow('Mice Detector', np.vstack([annotated, status]))

            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord('q'), KEY_ESC):
                break

            # frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print('Done.')


if __name__ == '__main__':
    main()
