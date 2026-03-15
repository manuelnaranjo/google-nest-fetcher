import argparse
import datetime
import os
import subprocess
import tempfile

import glocaltokens.client
import requests

FOYER_URL = (
    "https://googlehomefoyer-pa.clients6.google.com"
    "/$rpc/google.internal.home.foyer.v1.CameraService/GetHistoricalPlaybackUrl"
)
NEST_SCOPE = "oauth2:https://www.googleapis.com/auth/nest-account"
SEGMENT_MINUTES = 20 # experimental, the app and web allows for 5 minutes


def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_cameras(path):
    """Return a dict mapping friendly name -> device id."""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                device_id, name = parts
                cameras[name] = device_id
    return cameras


class NestAuthClient(glocaltokens.client.GLocalAuthenticationTokens):
    """Extends GLocalAuthenticationTokens to support fetching tokens for arbitrary scopes."""

    def get_access_token_for_service(self, service=NEST_SCOPE):
        master_token = self.get_master_token()
        res = glocaltokens.client.perform_oauth(
            self._escape_username(self.username),
            master_token,
            self.get_android_id(),
            app=glocaltokens.client.ACCESS_TOKEN_APP_NAME,
            service=service,
            client_sig=glocaltokens.client.ACCESS_TOKEN_CLIENT_SIGNATURE,
        )
        if "Auth" not in res:
            raise RuntimeError(f"Could not obtain access token: {res}")
        return res["Auth"]


def get_playback_url(access_token, device_id, start_ts, end_ts):
    payload = [None, device_id, [[int(start_ts)], [int(end_ts)]]]
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json+protobuf",
    }
    response = requests.post(FOYER_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()[0]


def download_segment(url, output_path, access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    tmp_path = output_path + ".tmp"
    try:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        os.replace(tmp_path, output_path)
    except:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Nest camera footage in 5-minute segments")
    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument("--device", help="Camera friendly name (from cameras.txt)")
    device_group.add_argument("--device-id", help="Camera device ID (raw UUID)")
    parser.add_argument("--start", required=True, help="Start time in ISO 8601 format (e.g. 2024-01-01T10:00:00)")
    parser.add_argument("--end", required=True, help="End time in ISO 8601 format")
    parser.add_argument("--output-dir", required=True, help="Directory to save downloaded footage")
    args = parser.parse_args()

    workspace = os.environ["BUILD_WORKSPACE_DIRECTORY"]
    env = load_env(os.path.join(workspace, ".env"))
    master_token = env.get("MASTER_TOKEN")
    if not master_token:
        parser.error("MASTER_TOKEN not found in .env file")
    username = env.get("USERNAME")
    if not username:
        parser.error("USERNAME not found in .env file")

    if args.device:
        cameras = load_cameras(os.path.join(workspace, "cameras.txt"))
        device_id = cameras[args.device]
    else:
        device_id = args.device_id

    start_dt = datetime.datetime.fromisoformat(args.start)
    end_dt = datetime.datetime.fromisoformat(args.end)
    if start_dt >= end_dt:
        parser.error("--start must be before --end")

    os.makedirs(args.output_dir, exist_ok=True)

    auth_client = NestAuthClient(master_token=master_token, username=username, password="FAKE_PASSWORD")
    access_token = auth_client.get_access_token_for_service()

    segments = []
    segment_delta = datetime.timedelta(minutes=SEGMENT_MINUTES)
    current = start_dt
    while current < end_dt:
        segment_end = min(current + segment_delta, end_dt)
        start_ts = int(current.timestamp())
        end_ts = int(segment_end.timestamp())

        filename = f"{current.strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(args.output_dir, filename)

        if os.path.exists(output_path):
            print(f"Skipping {current.isoformat()} -> {segment_end.isoformat()} (already exists)")
        else:
            print(f"Fetching {current.isoformat()} -> {segment_end.isoformat()} ...")
            video_url = get_playback_url(access_token, device_id, start_ts, end_ts)
            print(f"  Downloading -> {output_path}")
            download_segment(video_url, output_path, access_token)

        segments.append((current, segment_end, output_path))
        current = segment_end
