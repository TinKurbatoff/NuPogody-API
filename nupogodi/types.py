"""Value types for the NuPogodi environment.

These types contain *no* pygame or wall-clock dependencies. They describe the
game purely as data so the core logic, the Gymnasium wrapper, the renderer and
any remote transport can all agree on a single contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Side(IntEnum):
    """Which side of the screen the wolf's basket is on."""

    LEFT = 0
    RIGHT = 1


class Level(IntEnum):
    """Whether the basket is at the upper or lower chute of its side."""

    UP = 0
    DOWN = 1


class Quadrant(IntEnum):
    """One of the four chutes an egg can roll down / the basket can occupy.

    The integer encoding is ``side * 2 + level`` so it interoperates directly
    with :class:`Action` and the observation vector.
    """

    LEFT_UP = 0
    LEFT_DOWN = 1
    RIGHT_UP = 2
    RIGHT_DOWN = 3

    @classmethod
    def of(cls, side: Side, level: Level) -> Quadrant:
        return cls(int(side) * 2 + int(level))

    @property
    def side(self) -> Side:
        return Side(self // 2)

    @property
    def level(self) -> Level:
        return Level(self % 2)


class Action(IntEnum):
    """Absolute basket position the agent requests for the next tick.

    Identical encoding to :class:`Quadrant`; the two independent axes (side and
    level) are recovered as ``action // 2`` and ``action % 2``.
    """

    LEFT_UP = 0
    LEFT_DOWN = 1
    RIGHT_UP = 2
    RIGHT_DOWN = 3

    @property
    def side(self) -> Side:
        return Side(self // 2)

    @property
    def level(self) -> Level:
        return Level(self % 2)


@dataclass
class EggState:
    """A single egg rolling down a chute.

    ``state`` is the discrete animation/logic phase 0..9:

    * ``0..4`` — rolling down the chute (``4`` == at the catch point).
    * ``5..9`` — has fallen off the chute (only reachable once ``dropped``).

    ``dropped`` becomes ``True`` the tick the egg passes the catch point
    uncaught; the life has already been lost at that point and the remaining
    states are purely the falling animation.
    """

    quadrant: Quadrant
    state: int = 0
    dropped: bool = False


@dataclass
class GameState:
    """A complete, renderable snapshot of the game at one tick.

    Instances are value objects: two snapshots compare equal iff every field is
    equal, which is what the determinism test relies on. The core hands out
    *copies* so a stored snapshot never mutates underneath the caller.
    """

    wolf_side: Side
    wolf_level: Level
    eggs: list[EggState] = field(default_factory=list)
    score: int = 0
    lives: int = 3
    tick: int = 0

    @property
    def wolf_quadrant(self) -> Quadrant:
        return Quadrant.of(self.wolf_side, self.wolf_level)


@dataclass
class StepResult:
    """Everything one logical tick produced.

    ``caught``/``dropped``/``spawned`` are per-tick event counts used to drive
    reward, audio and logging without re-deriving them from the state diff.
    """

    state: GameState
    caught: int = 0
    dropped: int = 0
    spawned: int = 0

    @property
    def reward(self) -> int:
        return self.caught - self.dropped
