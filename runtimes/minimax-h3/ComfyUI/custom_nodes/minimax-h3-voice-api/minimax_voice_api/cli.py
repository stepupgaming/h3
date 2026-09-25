"""Command-line entry point for the API server."""
from __future__ import annotations

import argparse
import os

import uvicorn

from .config import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the MiniMax H3 Voice API")
    parser.add_argument("--host", default=os.environ.get("H3_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("H3_API_PORT", "8787")))
    parser.add_argument("--reload", action="store_true",
                        help="development only; restarts unload in-process state")
    args = parser.parse_args()
    uvicorn.run(
        "minimax_voice_api.app:app", host=args.host, port=args.port,
        reload=args.reload, timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
