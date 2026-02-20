import argparse
import os
import pprint
from glocaltokens.client import GLocalAuthenticationTokens


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List Google Home devices")
    parser.add_argument("username", help="Your Google username (email address)")
    args = parser.parse_args()

    workspace = os.environ["BUILD_WORKSPACE_DIRECTORY"]
    env = load_env(os.path.join(workspace, ".env"))
    master_token = env.get("MASTER_TOKEN")
    if not master_token:
        parser.error("MASTER_TOKEN not found in .env file")

    client = GLocalAuthenticationTokens(
        master_token=master_token,
        username=args.username,
        password="FAKE_PASSWORD"  # won't be used
    )

    homegraph_response = client.get_homegraph()

    for device in homegraph_response.home.devices:
        if device.device_type != 'action.devices.types.CAMERA':
            continue
        print(device.device_name)
        print(device.device_info)
