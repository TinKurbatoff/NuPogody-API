"""Pure, deterministic game logic for "Ну, погоди!" (Elektronika IM-02).

``NuPogodiCore`` owns the entire rule set with **no** pygame import and **no**
``time.time()``. Everything advances through :meth:`step`, one logical tick at a
time, and all randomness flows through a single seeded RNG so a ``(seed, action
sequence)`` pair reproduces a byte-for-byte identical trajectory.

Original mechanics preserved (see ``game.py`` in the repo history):

* Wolf basket occupies one of four quadrants; catching an egg at the catch
  point scores +1, missing one costs a life; the game ends at 0 of 3 lives.
* Difficulty scales with score: eggs roll faster (``score // 42``) and more of
  them may share a quadrant (``score // 15``), capped at five.

Judgment calls made while removing the wall clock are documented at each site.
"""

from __future__ import annotations

import random
from dataclasses import replace

from loguru import logger

from .types import Action, EggState, GameState, Level, Quadrant, Side, StepResult

# Egg-state milestones (0..9 phases; see EggState).
CATCH_STATE = 4  # at the basket; catchable this tick.
REMOVE_STATE = 10  # animation finished; egg leaves play.

MAX_LIVES = 3
MAX_EGGS_CAP = 5  # original `max_eggs`: hard ceiling on eggs per quadrant.

# Difficulty tuning, ported verbatim from the original module constants.
_SPEED = 1.05
_MIN_TICK_SECONDS = 0.5


class NuPogodiCore:
    """Headless, seedable game state machine.

    Parameters
    ----------
    dropped_advance:
        How many animation phases a *fallen* egg advances per tick. The original
        game advanced non-dropped eggs on the (score-dependent) main cadence but
        dropped eggs every 0.2 s — roughly 4-5x faster. Since a dropped egg has
        already cost its life and can no longer be interacted with, its fall is
        cosmetic; the only logical side effect is how long it keeps occupying a
        quadrant slot (which can briefly block a spawn). ``2`` gives a visible
        multi-tick fall while staying close to the original slot occupancy. This
        value never affects catches, drops, score or lives.
    """

    def __init__(self, dropped_advance: int = 2) -> None:
        self.dropped_advance = dropped_advance
        self._rng = random.Random()
        # Populate initial fields; reset() does the real work.
        self.wolf_side = Side.LEFT
        self.wolf_level = Level.UP
        self.eggs: list[EggState] = []
        self.score = 0
        self.lives = MAX_LIVES
        self.tick = 0
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: int | None = None) -> GameState:
        """Reset to a fresh game and (re)seed the RNG."""
        self._rng = random.Random(seed)
        self.wolf_side = Side.LEFT
        self.wolf_level = Level.UP
        self.eggs = []
        self.score = 0
        self.lives = MAX_LIVES
        self.tick = 0
        # trace, not debug: reset runs every episode in hot training loops.
        logger.trace("core reset (seed={})", seed)
        return self.state()

    def step(self, action: Action | int) -> StepResult:
        """Advance exactly one logical tick and report what happened.

        Order of operations (mirrors the original ``Egg.update`` which resolved
        catches/drops on the *current* state and only then advanced):

        1. apply the action (place the basket),
        2. resolve catches: a non-dropped egg at ``state == CATCH_STATE`` in the
           basket's quadrant is caught (+1 score, removed),
        3. resolve drops: any non-dropped egg already past the catch point costs
           a life and is flagged ``dropped``,
        4. advance every egg (non-dropped +1, dropped +``dropped_advance``),
        5. remove eggs whose animation has finished,
        6. attempt to spawn one new egg via the seeded RNG.

        Resolving *before* advancing (rather than the reverse) is deliberate: it
        keeps an egg visible at the catch point for a full tick before the catch
        is decided, which is what gives a human player their reaction window and
        makes the refactor indistinguishable from the original.
        """
        action = Action(int(action))

        # 1. apply action.
        self.wolf_side = action.side
        self.wolf_level = action.level
        basket = Quadrant.of(self.wolf_side, self.wolf_level)

        # 2. resolve catches.
        caught = 0
        survivors: list[EggState] = []
        for egg in self.eggs:
            if (
                not egg.dropped
                and egg.state == CATCH_STATE
                and egg.quadrant == basket
            ):
                caught += 1
                self.score += 1
            else:
                survivors.append(egg)
        self.eggs = survivors

        # 3. resolve drops (a present basket would have caught it at state 4, so
        # any non-dropped egg past the catch point is necessarily a miss).
        dropped = 0
        for egg in self.eggs:
            if not egg.dropped and egg.state > CATCH_STATE:
                egg.dropped = True
                dropped += 1
                self.lives -= 1

        # 4. advance.
        for egg in self.eggs:
            egg.state += self.dropped_advance if egg.dropped else 1

        # 5. remove finished eggs.
        self.eggs = [egg for egg in self.eggs if egg.state < REMOVE_STATE]

        # 6. spawn.
        spawned = self._spawn()

        self.tick += 1
        return StepResult(
            state=self.state(),
            caught=caught,
            dropped=dropped,
            spawned=spawned,
        )

    # -- derived difficulty -------------------------------------------------

    @property
    def current_max_eggs(self) -> int:
        """Max simultaneous eggs allowed per quadrant at the current score."""
        return min(max(1, self.score // 15 + 1), MAX_EGGS_CAP)

    def tick_seconds(self) -> float:
        """Real-time seconds a single tick should take at the current score.

        Pure function of ``score`` (no wall clock). The core never uses this;
        it exists so a real-time client (the human loop) can reproduce the
        original's score-driven speed-up without baking timing into the logic.
        """
        return max(1.0 / max(_SPEED ** (self.score // 42 + 1), 1.0), _MIN_TICK_SECONDS)

    @property
    def game_over(self) -> bool:
        return self.lives <= 0

    # -- snapshots ----------------------------------------------------------

    def state(self) -> GameState:
        """Return an independent, value-comparable snapshot of the game."""
        return GameState(
            wolf_side=self.wolf_side,
            wolf_level=self.wolf_level,
            eggs=[replace(egg) for egg in self.eggs],
            score=self.score,
            lives=self.lives,
            tick=self.tick,
        )

    # -- internals ----------------------------------------------------------

    def _spawn(self) -> int:
        """Attempt one egg spawn, honoring the per-quadrant cap.

        Faithful to the original ``summon_egg``: pick a random side then a
        random level (in that order, to keep RNG-consumption identical), and add
        an egg only if the chosen quadrant is below the cap. Dropped eggs still
        occupy their slot until removed, exactly as before.
        """
        left = self._rng.choice([True, False])
        up = self._rng.choice([True, False])
        quadrant = Quadrant.of(
            Side.LEFT if left else Side.RIGHT,
            Level.UP if up else Level.DOWN,
        )
        occupancy = sum(1 for egg in self.eggs if egg.quadrant == quadrant)
        if occupancy >= self.current_max_eggs:
            return 0
        self.eggs.append(EggState(quadrant=quadrant, state=0, dropped=False))
        return 1
