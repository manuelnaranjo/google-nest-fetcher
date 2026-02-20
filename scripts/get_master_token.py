import argparse
import json
import gpsoauth
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exchange token for master token")
    parser.add_argument("--email", default=os.environ.get("EMAIL"), help="Email address")
    parser.add_argument("--token", default=os.environ.get("TOKEN"), help="OAuth token")
    args = parser.parse_args()

    if not args.email or not args.token:
        parser.error("Both --email and --token must be provided via arguments or environment variables.")

    android_id = '0123456789abcdef' # leave this as is

    master_response = gpsoauth.exchange_token(args.email, args.token, android_id)

    print(f"Full response for debugging:")
    print(json.dumps(master_response))

    master_token = master_response['Token']
    print(f"\nMaster token {master_token}")
