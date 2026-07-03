"""Renderer smoke test.

Uses SDL's dummy video/audio drivers so the real (non-headless) render path —
asset loading and egg geometry for both chute and fallen states — is exercised
without a physical display. Skips cleanly if pygame or its assets are absent.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("pygame")

from nupogodi.core import NuPogodiCore  # noqa: E402
from nupogodi.renderer import PygameRenderer  # noqa: E402
from nupogodi.types import EggState, Quadrant  # noqa: E402


def test_headless_renderer_is_noop() -> None:
    r = PygameRenderer(headless=True)
    r.render(NuPogodiCore().state())  # must not raise or need a display
    r.close()


def test_renderer_draws_all_states() -> None:
    try:
        r = PygameRenderer(headless=False)
    except Exception as exc:  # missing display backend / assets
        pytest.skip(f"pygame display unavailable: {exc}")

    core = NuPogodiCore()
    core.reset(seed=42)
    core.eggs = [
        EggState(Quadrant.LEFT_UP, 2),
        EggState(Quadrant.RIGHT_DOWN, 4),
        EggState(Quadrant.LEFT_DOWN, 7, dropped=True),
        EggState(Quadrant.RIGHT_UP, 9, dropped=True),
    ]
    r.render(core.state(), sound_on=True)

    core.lives = 0  # game-over screen path
    r.render(core.state())
    r.close()
