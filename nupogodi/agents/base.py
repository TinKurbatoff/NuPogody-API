"""The agent contract.

An :class:`Agent` maps an observation to an action, and may optionally learn
from transitions. The same protocol is meant to be satisfied by a scripted
baseline, a silicon neural net (RL/SNN), or an adapter to a Cortical Labs CL1
biological network — none of which require the environment or core to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Transition:
    """One environment step, for agents that learn."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@runtime_checkable
class Agent(Protocol):
    """Anything that can choose an action given an observation."""

    def act(self, obs: np.ndarray) -> int:
        """Return an action in ``{0, 1, 2, 3}`` for the given observation."""
        ...

    def observe(self, transition: Transition) -> None:
        """Optionally consume a transition for learning. Default: no-op."""
        ...
