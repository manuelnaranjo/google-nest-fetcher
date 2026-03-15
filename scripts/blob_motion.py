"""Detect small moving blobs in a video and write an annotated output video."""

import argparse
import sys

import cv2


def draw_status_bar(frame, frame_num: int, blob_count: int):
    h, w = frame.shape[:2]
    bar_h = 24
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    text = f"frame={frame_num}  blobs={blob_count}"
    cv2.putText(frame, text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Detect small moving blobs in a video.")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--min-area", type=int, default=20, help="Min contour area in pixels (default: 20)")
    parser.add_argument("--max-area", type=int, default=2000, help="Max contour area in pixels (default: 2000)")
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size, must be odd (default: 5)")
    parser.add_argument("--history", type=int, default=500, help="MOG2 background history frames (default: 500)")
    parser.add_argument("--var-threshold", type=float, default=16.0, help="MOG2 variance threshold (default: 16)")
    parser.add_argument("--frame-skip", type=int, default=15, help="Run detection every N frames, reuse last frame otherwise (default: 15)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open {args.input}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: cannot open output {args.output}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    blur_k = args.blur | 1  # ensure odd
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=False,
    )
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    frame_num = 0
    last_out = None
    print(f"Processing {total_frames} frames -> {args.output}")

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_num % args.frame_skip == 0:
            _, frame = cap.retrieve()
            blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
            fg_mask = bg_sub.apply(blurred)
            fg_mask = cv2.erode(fg_mask, kernel_erode, iterations=1)
            fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=2)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            out = frame.copy()
            blob_count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if args.min_area <= area <= args.max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
                    blob_count += 1

            draw_status_bar(out, frame_num, blob_count)
            last_out = out

            writer.write(last_out if last_out is not None else frame)

        frame_num += 1
        if frame_num % 100 == 0:
            pct = frame_num / total_frames * 100 if total_frames > 0 else 0
            print(f"  {frame_num}/{total_frames} ({pct:.0f}%)", flush=True)

    cap.release()
    writer.release()
    print(f"Done. {frame_num} frames written to {args.output}")


if __name__ == "__main__":
    main()
