"""The core must be a pure function of (seed, action sequence).

No wall-clock dependence: two runs with the same seed and the same actions must
produce byte-for-byte identical GameState trajectories.
"""

from __future__ import annotations

import random

from nupogodi.core import NuPogodiCore
from nupogodi.types import Action, GameState


def _rollout(seed: int, actions: list[int]) -> list[GameState]:
    core = NuPogodiCore()
    core.reset(seed=seed)
    return [core.step(a).state for a in actions]


def _fixed_actions(n: int) -> list[int]:
    # Deterministic action script (its own seed), independent of the core RNG.
    rng = random.Random(1234)
    return [rng.randrange(4) for _ in range(n)]


def test_identical_seed_and_actions_reproduce_trajectory() -> None:
    actions = _fixed_actions(500)
    run_a = _rollout(42, actions)
    run_b = _rollout(42, actions)
    assert run_a == run_b


def test_different_seeds_diverge() -> None:
    actions = _fixed_actions(500)
    run_a = _rollout(42, actions)
    run_b = _rollout(7, actions)
    # Overwhelmingly likely to differ; if identical, seeding does nothing.
    assert run_a != run_b


def test_snapshots_are_independent_of_later_mutation() -> None:
    core = NuPogodiCore()
    core.reset(seed=3)
    first = core.step(Action.LEFT_UP).state
    snapshot = [
        (e.quadrant, e.state, e.dropped) for e in first.eggs
    ]
    # Advance many ticks; the earlier snapshot must not have changed.
    for _ in range(20):
        core.step(Action.RIGHT_DOWN)
    assert [(e.quadrant, e.state, e.dropped) for e in first.eggs] == snapshot


def test_reset_clears_state() -> None:
    core = NuPogodiCore()
    core.reset(seed=99)
    for _ in range(50):
        core.step(Action.LEFT_UP)
    fresh = core.reset(seed=99)
    assert fresh.score == 0
    assert fresh.lives == 3
    assert fresh.eggs == []
    assert fresh.tick == 0
