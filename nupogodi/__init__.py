"""NuPogodi — a headless, deterministic egg-catcher environment.

The package splits the classic single-file Pygame game into a clean
environment ↔ agent architecture:

* :mod:`nupogodi.core` — pure, seedable, tick-driven game logic.
* :mod:`nupogodi.env` — a Gymnasium wrapper (the agent contract).
* :mod:`nupogodi.renderer` — a read-only Pygame renderer.
* :mod:`nupogodi.clients` / :mod:`nupogodi.agents` — human and programmatic drivers.
"""

from __future__ import annotations

from .core import NuPogodiCore
from .env import NuPogodiEnv
from .types import (
    Action,
    EggState,
    GameState,
    Level,
    Quadrant,
    Side,
    StepResult,
)

__all__ = [
    "Action",
    "EggState",
    "GameState",
    "Level",
    "NuPogodiCore",
    "NuPogodiEnv",
    "Quadrant",
    "Side",
    "StepResult",
]

__version__ = "0.1.0"
