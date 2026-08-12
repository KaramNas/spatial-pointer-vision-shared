#!/usr/bin/env python
"""
Manual test client for server/main.py's /ws/identify endpoint. Sends a
single image file over the WebSocket exactly the way the Quest app's
InferenceClient.cs eventually will, and prints the response.

This exists to answer "is the server actually working" without needing
the Quest app to exist yet -- if this script gets a correct response, the
server side of the pipeline is proven; whatever sends bytes over this same
WebSocket protocol later (Quest or otherwise) will get the same result.

Usage (with server/main.py already running in another terminal):
    python server/test_client.py --image ../dataset/processed/dummy_images/dummy_000_red_mug.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets


async def send_image(uri: str, image_path: Path) -> None:
    image_bytes = image_path.read_bytes()

    async with websockets.connect(uri) as ws:
        print(f"[test_client] Connected to {uri}")
        print(f"[test_client] Sending {image_path.name} ({len(image_bytes)} bytes)")
        await ws.send(image_bytes)

        response_raw = await ws.recv()
        response = json.loads(response_raw)
        print(f"[test_client] Response: {json.dumps(response, indent=2)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image", type=Path, required=True, help="Path to a JPEG/PNG to send")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    uri = f"ws://{args.host}:{args.port}/ws/identify"
    asyncio.run(send_image(uri, args.image))


if __name__ == "__main__":
    main()
