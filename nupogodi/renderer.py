"""Read-only Pygame renderer for a :class:`~nupogodi.types.GameState`.

The renderer never mutates game state and never reads input — it is a pure
projection of a ``GameState`` onto the screen. The blitting (sprites, egg
coordinate tables, rotations, lives, score, background/grass/chicken/rabbit) is
ported verbatim from the original ``game.py`` so the picture is identical.

Coordinates are reconstructed from each egg's ``(quadrant, state)`` using the
same tables the original ``Egg`` precomputed per instance.
"""

from __future__ import annotations

import pathlib

from loguru import logger

from .types import GameState, Level, Quadrant, Side

_BASEMENT = pathlib.Path(__file__).parent.parent.absolute()
_IMAGES = _BASEMENT / "images"
_FONTS = _BASEMENT / "fonts"

WIDTH = 1000
HEIGHT = 630
_GRAY = (151, 151, 151)

# Chute coordinates for egg states 0..4 (from the original Egg tables).
_CHUTE_XY: dict[Quadrant, list[tuple[int, int]]] = {
    Quadrant.LEFT_UP: [(63, 178), (96, 197), (131, 228), (154, 249), (197, 258)],
    Quadrant.LEFT_DOWN: [(63, 324), (96, 343), (131, 364), (154, 395), (197, 404)],
    Quadrant.RIGHT_UP: [(918, 178), (881, 200), (840, 218), (805, 239), (775, 258)],
    Quadrant.RIGHT_DOWN: [(918, 324), (881, 343), (840, 364), (805, 385), (775, 404)],
}
# "Off the chute" coordinates for states 5..9 (the fall past the chicken).
_CHICKEN_XY_LEFT = [(160, 480), (132, 480), (96, 480), (62, 480), (38, 480)]
_CHICKEN_XY_RIGHT = [(720, 480), (760, 480), (826, 480), (860, 480), (920, 480)]

_LEFT_DEG = -30
_RIGHT_DEG = 30

_WOLF_IMAGES = {
    (Side.LEFT, Level.UP): "wolf_left_up.png",
    (Side.LEFT, Level.DOWN): "wolf_left_down.png",
    (Side.RIGHT, Level.UP): "wolf_right_up.png",
    (Side.RIGHT, Level.DOWN): "wolf_right_down.png",
}


class PygameRenderer:
    """Draws GameStates. Set ``headless=True`` to no-op (for training)."""

    def __init__(self, headless: bool = False) -> None:
        self.headless = headless
        self._pygame = None
        self._screen = None
        self._font = None
        self._end_font = None
        self._assets: dict[str, object] = {}
        if not headless:
            self._init_pygame()

    # -- setup -------------------------------------------------------------

    def _init_pygame(self) -> None:
        import pygame

        self._pygame = pygame
        pygame.init()
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ну, погоди!")
        self._load_assets()
        logger.debug("renderer initialised ({}x{})", WIDTH, HEIGHT)

    def _img(self, name: str):
        # Match the original: plain load + colorkey, no convert().
        pygame = self._pygame
        surface = pygame.image.load(str(_IMAGES / name))
        surface.set_colorkey(_GRAY)
        return surface

    def _load_assets(self) -> None:
        pygame = self._pygame
        a = self._assets
        a["background"] = self._img("background.png")
        a["grass"] = self._img("grass.png")
        a["right_chicken"] = self._img("right_chicken.png")
        a["life"] = self._img("life.png")
        a["rabbit_on"] = self._img("rabbit_on.png")
        a["rabbit_off"] = self._img("rabbit_off.png")
        for key, name in _WOLF_IMAGES.items():
            a[f"wolf_{int(key[0])}_{int(key[1])}"] = self._img(name)
        # egg_1..egg_6, plus horizontally-flipped copies for the right side.
        a["eggs_left"] = [self._img(f"egg_{i + 1}.png") for i in range(6)]
        a["eggs_right"] = [
            pygame.transform.flip(s, True, False) for s in a["eggs_left"]
        ]
        for s in a["eggs_right"]:
            s.set_colorkey(_GRAY)
        self._font = pygame.font.Font(str(_FONTS / "digital-7.ttf"), 72)
        self._end_font = pygame.font.Font(pygame.font.get_default_font(), 42)

    # -- egg geometry ------------------------------------------------------

    def _egg_blit(self, egg_quadrant: Quadrant, state: int):
        """Return (surface, (x, y)) for an egg, matching the original tables."""
        pygame = self._pygame
        a = self._assets
        idx = min(state, 9)
        is_left = egg_quadrant.side == Side.LEFT
        eggs = a["eggs_left"] if is_left else a["eggs_right"]

        if idx <= 4:
            xy = _CHUTE_XY[egg_quadrant][idx]
            surface = eggs[0]  # egg_1 while on the chute
        else:
            chicken = _CHICKEN_XY_LEFT if is_left else _CHICKEN_XY_RIGHT
            xy = chicken[idx - 5]
            surface = eggs[idx - 4]  # egg_2..egg_6 once fallen

        # Rotation only applies on the chute (state <= 4), as in the original.
        if state <= 4:
            deg = (_LEFT_DEG if is_left else _RIGHT_DEG) * state
            surface = pygame.transform.rotate(surface, deg)
            surface.set_colorkey(_GRAY)
        return surface, xy

    # -- rendering ---------------------------------------------------------

    def render(self, state: GameState, sound_on: bool = True) -> None:
        """Draw one frame from ``state``. No-op in headless mode.

        ``sound_on`` only selects the rabbit on/off indicator sprite; audio
        playback itself is the client's responsibility.
        """
        if self.headless:
            return
        pygame = self._pygame
        screen = self._screen
        a = self._assets

        screen.fill(_GRAY)

        if state.lives <= 0:
            self._render_game_over(state)
            pygame.display.flip()
            return

        screen.blit(a["background"], (0, 0))
        # Rabbit (sound indicator) then wolf, matching original draw order.
        screen.blit(a["rabbit_on"] if sound_on else a["rabbit_off"], (110, 0))
        wolf_key = f"wolf_{int(state.wolf_side)}_{int(state.wolf_level)}"
        screen.blit(a[wolf_key], (200, 220))

        for egg in state.eggs:
            surface, (x, y) = self._egg_blit(egg.quadrant, egg.state)
            screen.blit(surface, (x, y))

        screen.blit(a["right_chicken"], (0, 0))
        for i in range(abs(state.lives - 3)):
            screen.blit(a["life"], (500 + i * 80, 50))
        screen.blit(a["grass"], (0, 530))

        score_text = self._font.render(str(state.score), True, (20, 20, 20))
        screen.blit(score_text, (800, 50))

        pygame.display.flip()

    def _render_game_over(self, state: GameState) -> None:
        screen = self._screen
        text = self._end_font.render(
            f"Игра окончена! Счет: {state.score}", True, (20, 20, 20)
        )
        again = self._end_font.render(
            "Нажмите [SPACE] для повтора", True, (20, 20, 20)
        )
        rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        again_rect = again.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        screen.blit(text, rect)
        screen.blit(again, again_rect)

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
