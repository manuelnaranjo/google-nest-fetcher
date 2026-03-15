#!/usr/bin/env python3
"""
Visualize Mice Paths — Scans a video and generates a single summary image 
showing all movement paths detected throughout the footage.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

def get_background(video_path):
    """Extract a representative background image from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Try to get a frame from the middle where it might be stable
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames // 2, 100))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def main():
    parser = argparse.ArgumentParser(description="Visualize movement paths in a video.")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--output", default="path_summary.png", help="Output summary image (default: path_summary.png)")
    parser.add_argument("--frame-skip", type=int, default=5, help="Scan every Nth frame")
    parser.add_argument("--min-area", type=int, default=20, help="Min contour area for motion")
    parser.add_argument("--threshold", type=int, default=25, help="MOG2 variance threshold")
    parser.add_argument("--trail-color", default="0,255,0", help="BGR color for the path (default: 0,255,0)")
    
    args = parser.parse_args()
    
    color = tuple(map(int, args.trail_color.split(',')))
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        sys.exit(1)

    # 1. Get background
    background = get_background(args.input)
    if background is None:
        print("Error: Could not extract background frame.")
        sys.exit(1)
        
    h, w = background.shape[:2]
    path_layer = np.zeros_like(background)
    
    # 2. Setup Background Subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=args.threshold, detectShadows=False)
    
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Scanning {args.input}...")
    
    cv2.namedWindow("Mice Path Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mice Path Visualization", w, h)
    
    frame_idx = 0
    prev_display_frame = None
    paused = False
    
    # We'll use this to force a frame update when skipping while paused
    force_update = False
    
    while True:
        if not paused or force_update:
            if not force_update:
                for _ in range(args.frame_skip - 1):
                    if not cap.grab(): break
                    frame_idx += 1
                
            ret, frame = cap.read()
            if not ret: break
            
            if not force_update:
                frame_idx += 1
            force_update = False
            
            # Motion detection
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            fg_mask = bg_sub.apply(blurred)
            
            # Clean up
            fg_mask = cv2.erode(fg_mask, None, iterations=1)
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_detections = []
            motion_this_frame = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > args.min_area:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    center = (x + bw // 2, y + bh // 2)
                    cv2.circle(path_layer, center, 2, color, -1)
                    current_detections.append((x, y, bw, bh))
                    motion_this_frame = True
            
            # Base display is the current frame
            display = frame.copy()
            
            # Blend with previous frame if available for "ghosting" effect
            if prev_display_frame is not None:
                display = cv2.addWeighted(display, 0.6, prev_display_frame, 0.4, 0)
            
            # Highlight current detections with boxes
            for (x, y, bw, bh) in current_detections:
                cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            
            # Overlay cumulative path layer
            mask = cv2.cvtColor(path_layer, cv2.COLOR_BGR2GRAY) > 0
            path_overlay = display.copy()
            path_overlay[mask] = path_layer[mask]
            display = cv2.addWeighted(display, 0.5, path_overlay, 0.5, 0)
            
            prev_display_frame = frame.copy()
            
            # Draw progress text
            time_s = frame_idx / fps
            h_v, m_v, s_v = int(time_s // 3600), int((time_s % 3600) // 60), int(time_s % 60)
            time_str = f"{h_v:02d}:{m_v:02d}:{s_v:02d}"
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            
            pause_msg = " [PAUSED]" if paused else ""
            cv2.putText(display, f"Time: {time_str} ({pct:.1f}%){pause_msg}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "Space: Pause | Right: Skip 1s | Q: Quit", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            
            cv2.imshow("Mice Path Visualization", display)

        # Handle keys
        key = cv2.waitKeyEx(30 if paused else 1)
        if key in (ord('q'), 27): # q or Esc
            break
        elif key == 32: # Space
            paused = not paused
            # If we just paused, we already have the frame displayed.
            # If we unpause, the next loop will fetch a new frame.
            if paused:
                # Update display with [PAUSED] text
                cv2.putText(display, f"Time: {time_str} ({pct:.1f}%) [PAUSED]", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Mice Path Visualization", display)
        elif key in (65363, 83): # Right Arrow (Linux/X11 or generic)
            frame_idx = min(total_frames - 1, frame_idx + int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            force_update = True
        
        if not paused and frame_idx % 1000 < args.frame_skip:
            time_s = frame_idx / fps
            h_v, m_v, s_v = int(time_s // 3600), int((time_s % 3600) // 60), int(time_s % 60)
            time_str = f"{h_v:02d}:{m_v:02d}:{s_v:02d}"
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"  Progress: {time_str} / {total_frames} frames ({pct:.1f}%)", end="\r", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    print("\nScan complete.")
    
    # 3. Merge path layer with background
    # We use a bit of alpha blending to make the paths pop
    summary = cv2.addWeighted(background, 0.7, path_layer, 0.3, 0)
    
    # Alternatively, just overlay the paths where they are non-zero
    mask = cv2.cvtColor(path_layer, cv2.COLOR_BGR2GRAY) > 0
    summary = background.copy()
    summary[mask] = path_layer[mask]
    
    cv2.imwrite(args.output, summary)
    print(f"Path summary saved to {args.output}")

if __name__ == "__main__":
    main()
