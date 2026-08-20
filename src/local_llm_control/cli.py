from __future__ import annotations
import argparse
import asyncio
import json

import uvicorn

from .config import load_settings
from .manager import RuntimeManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage one local LLM runtime")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("profiles")
    sub.add_parser("status")
    start = sub.add_parser("start")
    start.add_argument("profile")
    sub.add_parser("stop")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    if args.action == "serve":
        uvicorn.run("local_llm_control.api:app", host=args.host, port=args.port)
        return

    manager = RuntimeManager(load_settings())
    if args.action == "profiles":
        result = manager.profiles()
    elif args.action == "status":
        result = manager.status()
    elif args.action == "start":
        result = asyncio.run(manager.start(args.profile))
    else:
        result = asyncio.run(manager.stop())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
