import argparse
import os
import sys

import requests


def send_onebot(message: str) -> bool:
    url = os.getenv("ONEBOT_HTTP_URL") or os.getenv("QQBOT_ONEBOT_URL")
    target = os.getenv("QQBOT_USER_ID") or os.getenv("QQBOT_GROUP_ID")
    if not url or not target:
        return False
    api = "send_private_msg" if os.getenv("QQBOT_USER_ID") else "send_group_msg"
    payload = {"message": message}
    if os.getenv("QQBOT_USER_ID"):
        payload["user_id"] = int(target)
    else:
        payload["group_id"] = int(target)
    token = os.getenv("ONEBOT_ACCESS_TOKEN") or os.getenv("QQBOT_ACCESS_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.post(f"{url.rstrip('/')}/{api}", json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return True


def send_webhook(message: str) -> bool:
    url = os.getenv("QQBOT_WEBHOOK_URL")
    if not url:
        return False
    resp = requests.post(url, json={"content": message, "message": message}, timeout=10)
    resp.raise_for_status()
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    args = parser.parse_args()
    try:
        if send_onebot(args.message) or send_webhook(args.message):
            print("notification sent")
            return 0
    except Exception as exc:
        print(f"notification failed: {exc}", file=sys.stderr)
        return 2
    print("notification skipped: set ONEBOT_HTTP_URL + QQBOT_USER_ID/QQBOT_GROUP_ID, or QQBOT_WEBHOOK_URL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
