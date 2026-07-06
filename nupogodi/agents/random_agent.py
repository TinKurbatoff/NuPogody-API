"""Baseline agent that samples uniformly random valid actions.

Doubles as a headless throughput benchmark: ``python -m
nupogodi.agents.random_agent`` runs a rollout and reports steps/sec.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ..env import NuPogodiEnv
from ..rollout import run
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
    """Run ``steps`` headless steps; return (episodes, steps_per_second).

    A thin wrapper over :func:`nupogodi.rollout.run` with no sinks, so it stays
    a pure throughput benchmark of the env+agent hot path.
    """
    stats = run(NuPogodiEnv(), RandomAgent(seed=seed), steps=steps, seed=seed)
    return stats.episodes, stats.steps_per_sec


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
