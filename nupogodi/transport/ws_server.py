"""WebSocket transport (STUB).

Two ways exist to drive the environment, and they serve different masters:

* **In-process** (``env.step`` directly) — the fast path. Zero serialization
  overhead, used for training silicon agents at tens/hundreds of thousands of
  steps per second.
* **WebSocket** (this module) — the remote path. Exposes ``reset``/``step`` over
  JSON so an out-of-process client (a human UI, a remote bot, or an adapter to a
  Cortical Labs CL1 biological network on separate hardware) can play over the
  network. Latency-bound, but decoupled from the Python process.

This is intentionally a scaffold: it fixes the wire protocol and the handler
shape without pulling in a websockets dependency yet. Fill in
:meth:`WebSocketEnvServer.serve` with a real async server when the remote path
is needed.

Wire protocol (JSON messages):
    -> {"cmd": "reset", "seed": 42}
    <- {"obs": [...], "info": {"score": 0, "lives": 3}}
    -> {"cmd": "step", "action": 2}
    <- {"obs": [...], "reward": 1.0, "terminated": false,
        "truncated": false, "info": {...}}
"""

from __future__ import annotations

from typing import Any

from ..env import NuPogodiEnv


class WebSocketEnvServer:
    """Translates JSON reset/step messages to env calls (transport-agnostic)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, **env_kwargs: Any):
        self.host = host
        self.port = port
        self.env = NuPogodiEnv(**env_kwargs)

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle one decoded JSON message and return the JSON-able reply.

        Pure and synchronous so it can be unit-tested without a socket; a real
        async ``serve`` loop would just decode frames and delegate here.
        """
        cmd = message.get("cmd")
        if cmd == "reset":
            obs, info = self.env.reset(seed=message.get("seed"))
            return {"obs": _as_list(obs), "info": _safe_info(info)}
        if cmd == "step":
            obs, reward, terminated, truncated, info = self.env.step(
                int(message["action"])
            )
            return {
                "obs": _as_list(obs),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": _safe_info(info),
            }
        raise ValueError(f"unknown command: {cmd!r}")

    async def serve(self) -> None:  # pragma: no cover - scaffold only.
        """Start the async WebSocket loop. Not implemented yet.

        Intended implementation: ``async with websockets.serve(...)`` accepting
        connections, decoding each JSON frame, calling :meth:`handle`, and
        sending the JSON reply back.
        """
        raise NotImplementedError(
            "WebSocket serving is a scaffold; use the in-process env.step path "
            "for training. Implement with the `websockets` package when the "
            "remote path is needed."
        )


def _as_list(obs: Any) -> list:
    return obs.tolist() if hasattr(obs, "tolist") else list(obs)


def _safe_info(info: dict[str, Any]) -> dict[str, Any]:
    """Drop the non-serializable raw GameState; keep the scalar summary."""
    return {k: v for k, v in info.items() if k != "state"}
