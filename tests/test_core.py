"""Rule-level tests for the pure core.

Spawning is RNG-driven and would add noise to hand-crafted lifecycle assertions,
so these tests disable ``_spawn`` and inject eggs directly to isolate the
catch / drop / removal logic. A separate test exercises spawning on its own.
"""

from __future__ import annotations

import pytest

from nupogodi.core import CATCH_STATE, REMOVE_STATE, NuPogodiCore
from nupogodi.types import Action, EggState, Quadrant


@pytest.fixture()
def core() -> NuPogodiCore:
    c = NuPogodiCore()
    c.reset(seed=0)
    return c


def _disable_spawn(c: NuPogodiCore) -> None:
    c._spawn = lambda: 0  # type: ignore[method-assign]


def test_catch_scores_and_removes_egg(core: NuPogodiCore) -> None:
    _disable_spawn(core)
    core.eggs = [EggState(quadrant=Quadrant.LEFT_UP, state=CATCH_STATE)]

    # Basket in the egg's quadrant on the tick it sits at the catch point.
    result = core.step(Action.LEFT_UP)

    assert result.caught == 1
    assert result.dropped == 0
    assert result.state.score == 1
    assert result.state.lives == 3
    assert result.state.eggs == []  # egg removed on catch


def test_miss_costs_a_life_one_tick_later(core: NuPogodiCore) -> None:
    _disable_spawn(core)
    core.eggs = [EggState(quadrant=Quadrant.LEFT_UP, state=CATCH_STATE)]

    # Wrong quadrant while the egg is catchable: no catch, egg advances to 5.
    r1 = core.step(Action.RIGHT_DOWN)
    assert r1.caught == 0
    assert r1.dropped == 0
    assert r1.state.lives == 3
    assert r1.state.eggs[0].state == CATCH_STATE + 1
    assert r1.state.eggs[0].dropped is False

    # Next tick it passes the catch point uncaught -> life lost, flagged dropped.
    r2 = core.step(Action.RIGHT_DOWN)
    assert r2.dropped == 1
    assert r2.state.lives == 2
    assert r2.state.eggs[0].dropped is True


def test_dropped_egg_is_removed_after_animation(core: NuPogodiCore) -> None:
    _disable_spawn(core)
    core.dropped_advance = 2
    core.eggs = [EggState(quadrant=Quadrant.LEFT_UP, state=CATCH_STATE + 1)]

    core.step(Action.RIGHT_UP)  # drop resolves: state 5 -> dropped, advance to 7
    assert core.eggs[0].dropped is True
    assert core.eggs[0].state == 7

    core.step(Action.RIGHT_UP)  # 7 -> 9
    assert core.eggs and core.eggs[0].state == 9

    core.step(Action.RIGHT_UP)  # 9 -> 11 >= REMOVE_STATE -> gone
    assert core.eggs == []
    assert REMOVE_STATE == 10


def test_full_lifecycle_roll_then_catch(core: NuPogodiCore) -> None:
    _disable_spawn(core)
    core.eggs = [EggState(quadrant=Quadrant.RIGHT_DOWN, state=0)]

    # Roll states 0->4 with the basket parked elsewhere; nothing scored/lost.
    for _ in range(CATCH_STATE):
        r = core.step(Action.LEFT_UP)
        assert r.caught == 0 and r.dropped == 0
    assert core.eggs[0].state == CATCH_STATE

    # Now move to the egg and catch it.
    r = core.step(Action.RIGHT_DOWN)
    assert r.caught == 1
    assert r.state.score == 1
    assert core.eggs == []


def test_game_over_at_zero_lives(core: NuPogodiCore) -> None:
    _disable_spawn(core)
    # Three eggs already past the catch point: each costs a life this tick.
    core.eggs = [
        EggState(quadrant=Quadrant.LEFT_UP, state=CATCH_STATE + 1),
        EggState(quadrant=Quadrant.LEFT_DOWN, state=CATCH_STATE + 1),
        EggState(quadrant=Quadrant.RIGHT_UP, state=CATCH_STATE + 1),
    ]
    result = core.step(Action.RIGHT_DOWN)
    assert result.dropped == 3
    assert result.state.lives == 0
    assert core.game_over is True


def test_spawn_respects_per_quadrant_cap() -> None:
    c = NuPogodiCore()
    c.reset(seed=1)
    # At score 0 the cap is 1 egg per quadrant.
    assert c.current_max_eggs == 1

    # Fill a quadrant to the cap, then assert no further egg lands there.
    c.eggs = [EggState(quadrant=q, state=0) for q in Quadrant]
    before = len(c.eggs)
    spawned = c._spawn()
    assert spawned == 0
    assert len(c.eggs) == before


def test_difficulty_curve_matches_original() -> None:
    c = NuPogodiCore()
    c.reset(seed=0)

    c.score = 0
    assert c.current_max_eggs == 1
    c.score = 15
    assert c.current_max_eggs == 2
    c.score = 200  # capped at 5
    assert c.current_max_eggs == 5

    # tick_seconds shrinks with score but never below the 0.5 s floor.
    c.score = 0
    assert c.tick_seconds() == pytest.approx(1.0 / 1.05, rel=1e-6)
    c.score = 10_000
    assert c.tick_seconds() == pytest.approx(0.5)
