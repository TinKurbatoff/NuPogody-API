"""The spiking CL1 agent: contract, learning wiring, checkpoint, integration.

Skipped wholesale when BindsNET isn't installed (the ``snn`` extra), mirroring
how the renderer tests lean on the SDL dummy driver — the core package stays
importable and testable without the optional heavy dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

# Import via the dish module: it installs the torch._six compat shim *before*
# importing bindsnet, so this skips only when bindsnet is genuinely absent
# (not because raw `import bindsnet` trips over modern torch).
pytest.importorskip("nupogodi.cl1.dish")

from nupogodi.agents.cl1_agent import CL1Agent  # noqa: E402
from nupogodi.cl1 import open as open_neurons  # noqa: E402
from nupogodi.env import NuPogodiEnv  # noqa: E402
from nupogodi.recorder import JsonlRecorder  # noqa: E402
from nupogodi.rollout import run  # noqa: E402


def _agent(**kw):
    # A tiny, fast culture keeps the tests quick.
    dish = open_neurons(backend="sim", neurons=32, seed=0)
    return CL1Agent(dish, window=4, seed=0, **kw)


def test_act_returns_a_valid_action_for_varied_observations():
    agent = _agent()
    env = NuPogodiEnv(flatten_obs=True)
    obs, _ = env.reset(seed=0)
    for probe in (obs, np.zeros(7, np.float32), np.ones(7, np.float32)):
        action = agent.act(probe)
        assert isinstance(action, int)
        assert 0 <= action <= 3


def test_observe_changes_plastic_weights_and_keeps_them_bounded():
    agent = _agent(reward_scale=20.0)
    dish = agent.neurons
    before = dish.state_dict()
    env = NuPogodiEnv(flatten_obs=True)
    obs, _ = env.reset(seed=0)
    from nupogodi.agents.base import Transition

    for _ in range(40):
        action = agent.act(obs)
        nxt, reward, term, trunc, info = env.step(action)
        agent.observe(Transition(obs, action, reward, nxt, term, trunc, info))
        obs = nxt if not (term or trunc) else env.reset()[0]

    after = dish.state_dict()
    assert not np.allclose(before["w_out"].numpy(), after["w_out"].numpy()), (
        "readout weights should move once reward-modulated STDP has run"
    )
    # Weights stay within the dish's configured bounds.
    assert float(after["w_out"].min()) >= 0.0
    assert float(after["w_in"].min()) >= 0.0


def test_learning_flag_disables_updates():
    agent = _agent()
    agent.learning = False
    before = agent.neurons.state_dict()["w_out"].clone()
    from nupogodi.agents.base import Transition

    obs = np.ones(7, np.float32)
    agent.act(obs)
    agent.observe(Transition(obs, 0, 1.0, obs, False, False, {}))
    after = agent.neurons.state_dict()["w_out"]
    assert np.allclose(before.numpy(), after.numpy())


def test_checkpoint_round_trip(tmp_path):
    agent = _agent(reward_scale=20.0)
    env = NuPogodiEnv(flatten_obs=True)
    run(env, agent, episodes=2, seed=0)  # learn something into critic + dish.
    path = str(tmp_path / "cl1.pt")
    agent.save(path)

    restored = _agent()
    restored.load(path)
    assert np.allclose(restored._critic, agent._critic)
    sd_a, sd_b = agent.neurons.state_dict(), restored.neurons.state_dict()
    assert np.allclose(sd_a["w_out"].numpy(), sd_b["w_out"].numpy())


def test_runs_through_rollout_and_recorder(tmp_path):
    agent = _agent()
    env = NuPogodiEnv(flatten_obs=True)
    log = tmp_path / "run.jsonl"
    with JsonlRecorder(path=str(log), meta={"agent": "cl1"}) as rec:
        stats = run(env, agent, episodes=2, seed=0, sinks=[rec])
    assert stats.episodes == 2
    assert stats.steps > 0
    lines = log.read_text().strip().splitlines()
    assert any('"type":"episode"' in ln for ln in lines)
    assert any('"type":"step"' in ln for ln in lines)
