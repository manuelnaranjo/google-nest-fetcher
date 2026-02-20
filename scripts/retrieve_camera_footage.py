import argparse
import datetime
import os

import glocaltokens.client
import requests

FOYER_URL = (
    "https://googlehomefoyer-pa.clients6.google.com"
    "/$rpc/google.internal.home.foyer.v1.CameraService/GetHistoricalPlaybackUrl"
)
NEST_SCOPE = "oauth2:https://www.googleapis.com/auth/nest-account"
SEGMENT_MINUTES = 5


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
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Nest camera footage in 5-minute segments")
    parser.add_argument("--username", required=True, help="Google username (email address)")
    parser.add_argument("--device-id", required=True, help="Camera device ID")
    parser.add_argument("--start", required=True, help="Start time in ISO 8601 format (e.g. 2024-01-01T10:00:00)")
    parser.add_argument("--end", required=True, help="End time in ISO 8601 format")
    parser.add_argument("--output-dir", required=True, help="Directory to save downloaded footage")
    args = parser.parse_args()

    workspace = os.environ["BUILD_WORKSPACE_DIRECTORY"]
    env = load_env(os.path.join(workspace, ".env"))
    master_token = env.get("MASTER_TOKEN")
    if not master_token:
        parser.error("MASTER_TOKEN not found in .env file")

    start_dt = datetime.datetime.fromisoformat(args.start)
    end_dt = datetime.datetime.fromisoformat(args.end)
    if start_dt >= end_dt:
        parser.error("--start must be before --end")

    os.makedirs(args.output_dir, exist_ok=True)

    auth_client = NestAuthClient(master_token=master_token, username=args.username, password="FAKE_PASSWORD")
    access_token = auth_client.get_access_token_for_service()

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
            current = segment_end
            continue

        print(f"Fetching {current.isoformat()} -> {segment_end.isoformat()} ...")
        video_url = get_playback_url(access_token, args.device_id, start_ts, end_ts)

        print(f"  Downloading -> {output_path}")
        download_segment(video_url, output_path, access_token)

        current = segment_end

    print("Done.")
