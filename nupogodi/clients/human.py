"""Human client — "the agent is a person at the keyboard".

This is the reference implementation of driving the env in real time while
preserving the exact feel of the original ``game.py``:

* The wolf moves *instantly* on a keypress. We render the basket at the player's
  latest requested position every frame, even though the env only *applies* that
  action on the next logical tick — so there's no perceived input lag.
* Egg motion and spawning happen once per logical tick, and the tick's real-time
  length is ``core.tick_seconds()`` — the same score-driven speed-up as the
  original.
* Because an egg stays at the catch point for a full tick before the catch is
  resolved, the player gets the same ~1 s reaction window as the original.

Controls (unchanged): WASD / arrows move the basket, ``1`` toggles sound,
``SPACE`` restarts at game over, ``ESC``/``Q`` quits.
"""

from __future__ import annotations

import pathlib

from loguru import logger

from ..env import NuPogodiEnv
from ..renderer import PygameRenderer
from ..types import Level, Side

_SOUNDS = pathlib.Path(__file__).parent.parent.parent.absolute() / "sounds"
_FPS = 30


def _action_of(side: Side, level: Level) -> int:
    return int(side) * 2 + int(level)


class HumanClient:
    def __init__(self) -> None:
        import pygame

        self.pygame = pygame
        self.env = NuPogodiEnv()
        self.renderer = PygameRenderer(headless=False)
        self.clock = pygame.time.Clock()

        pygame.mixer.init()
        self.sounds = {
            "ride": pygame.mixer.Sound(str(_SOUNDS / "egg_ride.ogg")),
            "crack": pygame.mixer.Sound(str(_SOUNDS / "egg_crack.ogg")),
            "catch": pygame.mixer.Sound(str(_SOUNDS / "egg_catch.ogg")),
        }
        self.sound_on = True

        # The player's currently-requested basket position (applied each tick).
        self.side = Side.LEFT
        self.level = Level.UP

    # -- input -------------------------------------------------------------

    def _handle_keydown(self, key) -> bool:
        """Update requested position / toggles. Returns False to quit."""
        pg = self.pygame
        if key in (pg.K_ESCAPE, pg.K_q):
            return False
        if key in (pg.K_LEFT, pg.K_a):
            self.side = Side.LEFT
        elif key in (pg.K_RIGHT, pg.K_d):
            self.side = Side.RIGHT
        elif key in (pg.K_UP, pg.K_w):
            self.level = Level.UP
        elif key in (pg.K_DOWN, pg.K_s):
            self.level = Level.DOWN
        elif key == pg.K_1:
            self.sound_on = not self.sound_on
        return True

    def _play(self, name: str) -> None:
        if self.sound_on:
            self.sounds[name].play()

    # -- loop --------------------------------------------------------------

    def run(self, seed: int | None = None) -> None:
        pg = self.pygame
        _, info = self.env.reset(seed=seed)
        state = info["state"]

        running = True
        accumulated = 0.0  # real seconds since the last logical tick.
        while running:
            dt = self.clock.tick(_FPS) / 1000.0

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if not self._handle_keydown(event.key):
                        running = False
                    if state.lives <= 0 and event.key == pg.K_SPACE:
                        _, info = self.env.reset()
                        state = info["state"]
                        accumulated = 0.0

            if state.lives > 0:
                accumulated += dt
                tick_len = self.env.core.tick_seconds()
                if accumulated >= tick_len:
                    accumulated -= tick_len
                    action = _action_of(self.side, self.level)
                    _, _, terminated, _, info = self.env.step(action)
                    state = info["state"]
                    if info["spawned"]:
                        self._play("ride")
                    if info["caught"]:
                        self._play("catch")
                    if info["dropped"]:
                        self._play("crack")
                    if terminated:
                        logger.info("game over, score={}", state.score)

            # Render the basket at the *requested* position for instant response.
            render_state = self._render_state(state)
            self.renderer.render(render_state, sound_on=self.sound_on)

        self.renderer.close()

    def _render_state(self, state):
        """Copy of the env state with the wolf at the player's live position."""
        from dataclasses import replace

        return replace(state, wolf_side=self.side, wolf_level=self.level)


def main() -> None:
    HumanClient().run()


if __name__ == "__main__":
    main()
