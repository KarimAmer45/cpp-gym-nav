"""Reference backend server for the socket bridge.

This process speaks the same newline-delimited JSON protocol a future Unreal
Engine process would implement, but is backed by the existing C++ core through
``NavEnv``. It is the *swappable backend*: replace this server with an Unreal
process that answers the identical protocol and the Python training code does
not change.

Protocol (newline-delimited JSON over TCP):

- On connect the server sends a handshake::

    {"protocol_version": 1, "observation_low": [...], "observation_high": [...],
     "action_low": [...], "action_high": [...]}

- Client -> server requests, one JSON object per line::

    {"cmd": "reset", "seed": <int|null>}
    {"cmd": "step", "action": [linear, angular]}   # normalized in [-1, 1]
    {"cmd": "close"}

- Server -> client responses::

    reset: {"observation": [...], "info": {...}}
    step:  {"observation": [...], "reward": r, "terminated": b,
            "truncated": b, "info": {...}}
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from cpp_gym_nav.nav_env import NavEnv, NavEnvConfig

PROTOCOL_VERSION = 1


def _jsonable(info: dict[str, Any]) -> dict[str, Any]:
    """Coerce NumPy scalars/arrays in an info dict to JSON-native types."""
    out: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = float(value)
        elif isinstance(value, np.bool_):
            out[key] = bool(value)
        else:
            out[key] = value
    return out


def handle_connection(conn: socket.socket, config: NavEnvConfig) -> None:
    """Serve one client connection with a dedicated environment instance."""
    env = NavEnv(config=config)
    reader = conn.makefile("r", encoding="utf-8")

    def send(payload: dict[str, Any]) -> None:
        conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    try:
        send(
            {
                "protocol_version": PROTOCOL_VERSION,
                "observation_low": env.observation_space.low.tolist(),
                "observation_high": env.observation_space.high.tolist(),
                "action_low": env.action_space.low.tolist(),
                "action_high": env.action_space.high.tolist(),
            }
        )
        for line in reader:
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)
            command = message.get("cmd")
            if command == "reset":
                observation, info = env.reset(seed=message.get("seed"))
                send({"observation": observation.tolist(), "info": _jsonable(info)})
            elif command == "step":
                action = np.asarray(message["action"], dtype=np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                send(
                    {
                        "observation": observation.tolist(),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": _jsonable(info),
                    }
                )
            elif command == "close":
                break
            else:
                send({"error": f"unknown command: {command!r}"})
    finally:
        reader.close()
        env.close()
        conn.close()


def start_reference_server(
    config: NavEnvConfig | None = None,
) -> tuple[str, int, Callable[[], None]]:
    """Start an in-process single-client reference server on an ephemeral port.

    Returns ``(host, port, stop)``; ``stop()`` closes the listener and joins the
    thread. Used by ``NavEnvUnreal`` so the bridge is self-contained for tests
    and ``check_env`` without launching a separate process.
    """
    resolved = config or NavEnvConfig()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()

    def run() -> None:
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        handle_connection(conn, resolved)

    thread = threading.Thread(target=run, name="nav-reference-backend", daemon=True)
    thread.start()

    def stop() -> None:
        try:
            listener.close()
        except OSError:
            pass
        thread.join(timeout=1.0)

    return host, port, stop


def serve(host: str, port: int, config: NavEnvConfig | None = None) -> None:
    """Run a standalone multi-client server (one environment per connection)."""
    resolved = config or NavEnvConfig()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen()
        print(f"reference backend listening on {host}:{port} (protocol v{PROTOCOL_VERSION})")
        while True:
            conn, address = listener.accept()
            print(f"client connected: {address}")
            threading.Thread(target=handle_connection, args=(conn, resolved), daemon=True).start()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="cpp-gym-nav reference socket backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8917)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
