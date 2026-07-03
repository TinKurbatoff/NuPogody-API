"""Baseline agent that samples uniformly random valid actions.

Doubles as a headless throughput benchmark: ``python -m
nupogodi.agents.random_agent`` runs a rollout and reports steps/sec.
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

from ..env import NuPogodiEnv
from .base import Transition


class RandomAgent:
    """Samples a random action; ignores the observation."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(4))

    def observe(self, transition: Transition) -> None:  # no learning.
        return None


def rollout(steps: int = 100_000, seed: int = 0) -> tuple[int, float]:
    """Run ``steps`` headless steps; return (episodes, steps_per_second)."""
    env = NuPogodiEnv()
    agent = RandomAgent(seed=seed)
    obs, _ = env.reset(seed=seed)

    episodes = 0
    start = time.perf_counter()
    for _ in range(steps):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.observe(
            Transition(obs, action, reward, obs, terminated, truncated, info)
        )
        if terminated or truncated:
            episodes += 1
            obs, _ = env.reset()
    elapsed = time.perf_counter() - start

    steps_per_sec = steps / elapsed if elapsed else float("inf")
    return episodes, steps_per_sec


def main() -> None:
    steps = 100_000
    episodes, sps = rollout(steps=steps)
    logger.info(
        "random-agent rollout: {} steps, {} episodes, {:.0f} steps/sec",
        steps,
        episodes,
        sps,
    )
    print(f"{steps} steps over {episodes} episodes -> {sps:,.0f} steps/sec")


if __name__ == "__main__":
    main()
