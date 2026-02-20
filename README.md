# home-fetcher

Download historical footage from Google Nest cameras.

Based on [google-nest-telegram-sync](https://github.com/TamirMa/google-nest-telegram-sync)

I use this script mostly to retrieve long videos of my childs camera, he suffers from a very rare
genetic syndrome and had sleep apneas with many incidents where he even stops to breath. Google
nest cameras are OK for this as they detect events, or allow to download videos in increments
of 5 minutes through the app or web. But I need longer footage to share with the medical staff.

Also turns out the non documented API this tool uses allows for 20 minutes at once.

Take into account that 20 minutes of recording are roughly 200MB.

## Prerequisites

- [Bazel](https://bazel.build/) installed
- `ffmpeg` available on your `PATH` (used for merging segments)
- [`direnv`](https://direnv.net/) installed and configured in case you're going to develop
- A Google account with Nest cameras linked to Google Home
- A Google Home subscription with 24/7 recording enabled (required to access historical footage stored in the cloud)

## Step 1 — Get a master token

Follow the steps in [this gist](https://gist.github.com/julianpitt/94774e74d36a46f9ec10ddce13cfd423) to obtain an OAuth2 token for your Google account.

Then exchange it for a long-lived master token:

```sh
EMAIL=<your-email> TOKEN=<your-oauth2-token> bazel run :get_master_token
```

Copy the `Master token` value from the output and add it to a `.env` file in the repo root:

```sh
MASTER_TOKEN="aas_et/..."
```

## Step 2 — Find your camera's device ID

List all Nest cameras linked to your account:

```sh
bazel run :list_cameras -- <your-email>
```

The output prints each camera's name and device info. Note the `unique_id` field from the device info — that is the `device-id` used in the next step.

## Step 3 — Download footage

```sh
bazel run :retrieve_camera_footage -- \
  --username <your-email> \
  --device-id <device-id> \
  --start 2024-01-01T10:00:00 \
  --end   2024-01-01T11:00:00 \
  --output-dir /path/to/output
```

> **Note on timezones:** `--start` and `--end` are interpreted in the local timezone of the machine running the script, which may differ from the timezone where the camera is located. Adjust accordingly.

The script downloads footage in 20-minute segments into `--output-dir`. Segments that were already downloaded on a previous run are skipped automatically.

Once all segments are present, they are concatenated into a single MP4 file named after the requested time range (e.g. `20240101_100000_to_20240101_110000.mp4`). The merged file includes a subtitle track with the wall-clock timestamp of each segment.
