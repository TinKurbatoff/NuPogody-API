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
from .recorder import JsonlRecorder
from .rollout import EpisodeSummary, RolloutStats, Sink, run
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
    "EpisodeSummary",
    "GameState",
    "JsonlRecorder",
    "Level",
    "NuPogodiCore",
    "NuPogodiEnv",
    "Quadrant",
    "RolloutStats",
    "Side",
    "Sink",
    "StepResult",
    "run",
]

__version__ = "0.1.0"
