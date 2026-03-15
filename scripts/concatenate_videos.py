import argparse
import datetime
import os
import subprocess
import tempfile
from time import strftime

def _srt_timestamp(td):
    total_ms = int(td.total_seconds() * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

def concatenate_segments(segments, output_path):
    """Concatenate segment mp4 files with an SRT subtitle track showing wall-clock time.

    segments: list of file paths in order.
    """

    # Build ffmpeg concat list and SRT in one pass.
    concat_lines = []
    srt_lines = []
    video_cursor = datetime.timedelta()
    for i in range(1, len(segments)):
        path = segments[i]
        prev_path = segments[i - 1]

        prev_name = os.path.splitext(os.path.basename(prev_path))[0]
        curr_name = os.path.splitext(os.path.basename(path))[0]

        seg_start = datetime.datetime.strptime(prev_name, "%Y%m%d_%H%M%S")
        seg_end = datetime.datetime.strptime(curr_name, "%Y%m%d_%H%M%S")

        concat_lines.append(f"file '{os.path.abspath(path)}'")

        seg_duration = seg_end - seg_start
        srt_lines.append(str(i))
        srt_lines.append(
            f"{_srt_timestamp(video_cursor)} --> {_srt_timestamp(video_cursor + seg_duration)}"
        )
        srt_lines.append(seg_start.strftime("%Y-%m-%d %H:%M:%S"))
        srt_lines.append("")
        video_cursor += seg_duration

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(concat_lines))
        concat_list_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
        f.write("\n".join(srt_lines))
        srt_path = f.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-i", srt_path,
                "-c", "copy",
                "-c:s", "mov_text",
                output_path,
            ],
            check=True,
        )
    finally:
        os.unlink(concat_list_path)
        os.unlink(srt_path)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate Nest camera footage in 5-minute segments")
    parser.add_argument("--videos", required=True, nargs="+", help="Directory containing downloaded footage to concatenate")
    parser.add_argument("--output", required=True, help="Output file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Concatenating segments ...")
    merged_path = concatenate_segments(sorted(args.videos), args.output)
    print(f"Done. Output: {merged_path}")
