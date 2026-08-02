"""Gymnasium environment driven over a socket, for an Unreal (or other) backend.

``NavEnvUnreal`` satisfies the *identical* Gymnasium contract as ``NavEnv`` but
delegates simulation to a separate process over a versioned TCP protocol (see
``backend_server``). This is the "Unreal Engine + Gymnasium API" integration
pattern: the Python training code is unchanged whether the backend is the local
C++ core (the reference server) or an Unreal Engine process answering the same
protocol.

With no ``host``/``port`` it launches an in-process reference backend (the C++
core), so it is self-contained for ``check_env`` and tests. Point it at a running
Unreal server with ``NavEnvUnreal(host, port)``.
"""

from __future__ import annotations

import json
import socket
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cpp_gym_nav.backend_server import PROTOCOL_VERSION, start_reference_server
from cpp_gym_nav.nav_env import NavEnvConfig


class NavEnvUnreal(gym.Env[np.ndarray, np.ndarray]):
    """Navigation environment whose backend is reached over a socket."""

    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        *,
        config: NavEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None:
            raise ValueError("NavEnvUnreal does not stream frames; render on the backend side")
        self.render_mode = None

        self._stop = None
        if host is None or port is None:
            # Launch the reference backend (C++ core) in-process for a
            # self-contained, testable bridge over a real TCP socket.
            host, port, self._stop = start_reference_server(config)

        self._sock = socket.create_connection((host, port))
        self._reader = self._sock.makefile("r", encoding="utf-8")
        self._closed = False

        handshake = json.loads(self._reader.readline())
        if handshake.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(
                f"backend protocol version {handshake.get('protocol_version')} != "
                f"expected {PROTOCOL_VERSION}"
            )
        self.observation_space = spaces.Box(
            low=np.asarray(handshake["observation_low"], dtype=np.float32),
            high=np.asarray(handshake["observation_high"], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray(handshake["action_low"], dtype=np.float32),
            high=np.asarray(handshake["action_high"], dtype=np.float32),
            dtype=np.float32,
        )

    def _rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        self._sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        line = self._reader.readline()
        if not line:
            raise ConnectionError("backend closed the connection")
        return json.loads(line)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        response = self._rpc({"cmd": "reset", "seed": seed})
        observation = np.asarray(response["observation"], dtype=np.float32)
        return observation, dict(response.get("info", {}))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        response = self._rpc({"cmd": "step", "action": action.tolist()})
        observation = np.asarray(response["observation"], dtype=np.float32)
        return (
            observation,
            float(response["reward"]),
            bool(response["terminated"]),
            bool(response["truncated"]),
            dict(response.get("info", {})),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.sendall((json.dumps({"cmd": "close"}) + "\n").encode("utf-8"))
        except OSError:
            pass
        try:
            self._reader.close()
            self._sock.close()
        except OSError:
            pass
        if self._stop is not None:
            self._stop()
