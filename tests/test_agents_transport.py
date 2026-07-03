"""Tests for the agent protocol, random agent, and WebSocket transport stub."""

from __future__ import annotations

import numpy as np
import pytest

from nupogodi.agents import Agent, RandomAgent
from nupogodi.agents.random_agent import rollout
from nupogodi.transport.ws_server import WebSocketEnvServer


def test_random_agent_satisfies_protocol() -> None:
    agent = RandomAgent(seed=0)
    assert isinstance(agent, Agent)
    action = agent.act(np.zeros(7))
    assert action in {0, 1, 2, 3}


def test_random_agent_is_seeded() -> None:
    a = RandomAgent(seed=123)
    b = RandomAgent(seed=123)
    obs = np.zeros(7)
    assert [a.act(obs) for _ in range(20)] == [b.act(obs) for _ in range(20)]


def test_rollout_reports_throughput() -> None:
    episodes, sps = rollout(steps=2000, seed=1)
    assert episodes > 0
    assert sps > 0


def test_ws_handle_reset_and_step() -> None:
    server = WebSocketEnvServer()
    reply = server.handle({"cmd": "reset", "seed": 42})
    assert "obs" in reply
    assert reply["info"]["lives"] == 3
    assert "state" not in reply["info"]  # GameState is stripped for the wire

    reply = server.handle({"cmd": "step", "action": 0})
    assert set(reply) == {"obs", "reward", "terminated", "truncated", "info"}
    assert isinstance(reply["reward"], float)


def test_ws_rejects_unknown_command() -> None:
    server = WebSocketEnvServer()
    with pytest.raises(ValueError):
        server.handle({"cmd": "explode"})
