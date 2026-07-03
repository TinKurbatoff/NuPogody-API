"""Gymnasium wrapper around :class:`~nupogodi.core.NuPogodiCore`.

The env is the stable *contract* between the game and any agent — human, silicon
neural net, or a biological Cortical Labs CL1 network — none of which require
changing the core.

Spec
----
* **Action** — ``Discrete(4)``: ``0=left-up, 1=left-down, 2=right-up,
  3=right-down`` (absolute basket position).
* **Observation** — by default ``MultiDiscrete([2, 2, 10, 10, 10, 10, 4])``:
  ``[wolf_side, wolf_level, q_LU, q_LD, q_RU, q_RD, lives]`` where each ``q_*``
  is the state (0..9) of the *nearest-to-catch* non-dropped egg in that
  quadrant, or ``0`` if the quadrant is empty. With ``flatten_obs=True`` the
  same information is returned as a normalized ``Box`` float vector of shape
  ``(7,)`` for neural-network convenience.
* **Reward** — ``+1`` per egg caught this tick, ``-1`` per life lost this tick.
* **terminated** — ``lives <= 0``. **truncated** — optional ``max_steps`` reached.
* ``info["state"]`` — the raw :class:`~nupogodi.types.GameState` for rendering.

Caveat on the observation: encoding an empty quadrant as ``0`` collides with a
freshly-spawned egg sitting at ``state 0``. That is intentional and harmless —
a just-spawned egg is maximally far from the catch point, so treating it as
"nothing to react to yet" loses no actionable signal.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import NuPogodiCore
from .types import GameState, Quadrant


def _quadrant_signal(state: GameState) -> list[int]:
    """Nearest-to-catch non-dropped egg state per quadrant (0 if empty)."""
    signal = [0, 0, 0, 0]
    for egg in state.eggs:
        if egg.dropped:
            continue
        # "Nearest" = highest state = closest to the catch point.
        if egg.state > signal[int(egg.quadrant)]:
            signal[int(egg.quadrant)] = egg.state
    return signal


class NuPogodiEnv(gym.Env):
    """Deterministic, headless-friendly Gymnasium env for the egg catcher."""

    metadata = {"render_modes": [], "render_fps": 30}

    #: MultiDiscrete component sizes: side, level, 4 quadrants, lives.
    _NVEC = np.array([2, 2, 10, 10, 10, 10, 4], dtype=np.int64)

    def __init__(
        self,
        *,
        flatten_obs: bool = False,
        max_steps: int | None = None,
        reward_shaping: bool = False,
        dropped_advance: int = 2,
    ) -> None:
        super().__init__()
        self.core = NuPogodiCore(dropped_advance=dropped_advance)
        self.flatten_obs = flatten_obs
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self._elapsed_steps = 0

        self.action_space = spaces.Discrete(4)
        if flatten_obs:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(7,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.MultiDiscrete(self._NVEC)

    # -- gym API -----------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        state = self.core.reset(seed=seed)
        self._elapsed_steps = 0
        return self._observe(state), self._info(state)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self.core.step(action)
        self._elapsed_steps += 1

        reward = float(result.reward)
        if self.reward_shaping:
            reward = self._shape_reward(reward, result.state)

        terminated = self.core.game_over
        truncated = self.max_steps is not None and self._elapsed_steps >= self.max_steps

        info = self._info(result.state)
        info.update(
            caught=result.caught,
            dropped=result.dropped,
            spawned=result.spawned,
        )
        return self._observe(result.state), reward, terminated, truncated, info

    # -- helpers -----------------------------------------------------------

    def _observe(self, state: GameState) -> np.ndarray:
        q = _quadrant_signal(state)
        raw = [
            int(state.wolf_side),
            int(state.wolf_level),
            q[Quadrant.LEFT_UP],
            q[Quadrant.LEFT_DOWN],
            q[Quadrant.RIGHT_UP],
            q[Quadrant.RIGHT_DOWN],
            max(0, state.lives),
        ]
        if not self.flatten_obs:
            return np.array(raw, dtype=np.int64)
        # Normalize each component into [0, 1] for NN-friendly input.
        norm = np.array(
            [
                raw[0] / 1.0,
                raw[1] / 1.0,
                raw[2] / 9.0,
                raw[3] / 9.0,
                raw[4] / 9.0,
                raw[5] / 9.0,
                raw[6] / 3.0,
            ],
            dtype=np.float32,
        )
        return norm

    def _info(self, state: GameState) -> dict[str, Any]:
        return {"state": state, "score": state.score, "lives": state.lives}

    def _shape_reward(self, reward: float, state: GameState) -> float:
        """Optional shaping hook (disabled by default).

        Kept intentionally minimal: a tiny survival bonus. Off unless the caller
        opts in, so the default reward matches the raw game outcome exactly.
        """
        return reward + 0.01

    def render(self) -> None:  # rendering lives in PygameRenderer, driven by info.
        return None

    def close(self) -> None:
        return None
