"""Gymnasium-contract tests for NuPogodiEnv."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from nupogodi.env import NuPogodiEnv


def test_passes_gym_env_checker() -> None:
    env = NuPogodiEnv()
    # skip_render_check: rendering is handled out-of-band by PygameRenderer.
    check_env(env, skip_render_check=True)


def test_flatten_obs_passes_env_checker() -> None:
    env = NuPogodiEnv(flatten_obs=True)
    check_env(env, skip_render_check=True)


def test_reset_is_reproducible() -> None:
    env = NuPogodiEnv()
    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)


def test_info_carries_raw_state() -> None:
    env = NuPogodiEnv()
    _, info = env.reset(seed=0)
    assert "state" in info
    _, _, _, _, info = env.step(0)
    from nupogodi.types import GameState

    assert isinstance(info["state"], GameState)
    assert {"caught", "dropped", "spawned"} <= info.keys()


def test_random_agent_10k_steps_headless() -> None:
    env = NuPogodiEnv()
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)

    episodes = 0
    total_reward = 0.0
    for _ in range(10_000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        total_reward += reward
        if terminated or truncated:
            episodes += 1
            env.reset()

    # A random agent loses lives, so episodes must actually end and restart.
    assert episodes > 0


def test_truncation_on_max_steps() -> None:
    env = NuPogodiEnv(max_steps=5)
    env.reset(seed=0)
    truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated:
            break
    assert truncated is True
