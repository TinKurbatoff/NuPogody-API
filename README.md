# Ну, Погоди! — API Edition

**This fork makes the game controllable through an API.** Instead of (or in addition to) keyboard input, the game can be driven programmatically — moving the wolf, toggling sound, restarting the game and reading the game state (score, misses, egg positions) from external code. This makes it suitable for bots, automation experiments and reinforcement-learning agents.

This is a fork of [Ar4ikov/NuPogody](https://github.com/Ar4ikov/NuPogody).

«Ну, погоди!» («Электроника ИМ-02») — электронная игра, самая известная и популярная из серии первых советских портативных электронных игр с жидкокристаллическим экраном, производимых под торговой маркой «Электроника». Производилась с 1984 года. Кроме игры, устройство обладает функцией часов и будильника. Розничная цена составляла 25 рублей (позднее — 23 рубля).
Первоначально не имела категорийного номера. Впоследствии получила номер «Электроника ИМ-02». Аббревиатура «ИМ» означает «игра микропроцессорная».
Микропроцессор: КБ1013ВК1-2, дисплей ИЖМ2-71-01 (в первых выпусках) или ИЖМ13-71.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python game.py
```

## Управление

* **A** — влево
* **D** — вправо
* **W** — вверх
* **S** — вниз
* **1** - Включить/выключить звук
* **ENTER** - перезапуск

## Скриншоты

![Image1](presentation/presentation-1.png)
![Image2](presentation/presentation-2.png)

---

# `nupogodi` — headless environment + swappable agents

The original `game.py` fused input, game logic, rendering and audio into one
Pygame loop driven by wall-clock time. The `nupogodi/` package refactors that
into a clean **environment ↔ agent** architecture built around the
[Gymnasium](https://gymnasium.farama.org/) interface, so the *exact same game
core* can be driven by a human, a silicon neural net, or a Cortical Labs CL1
biological network — without changing the core. Game rules and on-screen
behavior are unchanged; only the concerns are decoupled.

The core is **deterministic and tick-driven**: everything advances through
`step()`, all randomness flows through one seedable RNG, and there is no
`time.time()` anywhere in the logic. Given a seed and an action sequence, the
trajectory is reproducible byte-for-byte.

```
nupogodi/
  core.py         # NuPogodiCore: pure logic + state. No pygame, no time.time().
  types.py        # Side/Level/Quadrant/Action enums, EggState/GameState dataclasses.
  env.py          # NuPogodiEnv(gymnasium.Env) wrapping the core.
  renderer.py     # PygameRenderer: read-only, draws a GameState. Reuses assets.
  clients/human.py    # keyboard -> action -> env.step -> render (reference "human agent").
  agents/             # Agent protocol + random baseline (RL/SNN/CL1 plug in here).
  transport/ws_server.py  # WebSocket wrapper (stub) for remote/human/bio clients.
tests/              # core, determinism, env-checker, renderer, agents/transport.
```

## Install

```bash
pip install -e '.[dev]'      # core + tests + ruff + pygame
pip install -e '.[render]'   # add pygame only (for the human client)
```

## Environment spec

| Aspect | Value |
| --- | --- |
| **Action** | `Discrete(4)` — `0=left-up, 1=left-down, 2=right-up, 3=right-down` (absolute basket position) |
| **Observation** (default) | `MultiDiscrete([2, 2, 10, 10, 10, 10, 4])` = `[wolf_side, wolf_level, q_LU, q_LD, q_RU, q_RD, lives]` |
| **Observation** (`flatten_obs=True`) | `Box(0, 1, shape=(7,), float32)` — the same vector, normalized, for NN input |
| **Reward** | `+1` per egg caught this tick, `-1` per life lost this tick |
| **terminated** | `lives <= 0` |
| **truncated** | optional `max_steps` reached |
| **info["state"]** | the raw `GameState` (so the renderer draws without re-deriving anything) |

Each `q_*` is the state (`0..9`) of the nearest-to-catch non-dropped egg in that
quadrant, or `0` if empty. Constructor flags: `flatten_obs`, `max_steps`,
`reward_shaping` (off by default), `dropped_advance`.

## Running it

**(a) Human client** — play with the keyboard (WASD/arrows move, `1` toggles
sound, `SPACE` restarts, `ESC`/`Q` quits):

```bash
python -m nupogodi.clients.human      # or: nupogodi-human
```

**(b) Headless random-agent rollout** — no window, no audio, prints steps/sec:

```bash
python -m nupogodi.agents.random_agent   # or: nupogodi-random
```

In-process training loop:

```python
from nupogodi import NuPogodiEnv

env = NuPogodiEnv(flatten_obs=True, max_steps=1000)
obs, info = env.reset(seed=42)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
```

## Tests & performance

```bash
pytest          # core rules, determinism, gymnasium env-checker, renderer, agents
ruff check .
```

The env runs fully headless (no display, no audio). On this machine the random
agent sustains **~100,000 steps/sec** (measured over 200k steps / ~15,600
episodes; also passes `gymnasium.utils.env_checker.check_env` and 10k steps with
no exceptions).

## Judgment calls when removing the wall clock

* **Resolve-then-advance ordering.** The original resolved a catch/miss on an
  egg's *current* state and only then advanced it. `step()` keeps that order
  (catches → drops → advance), rather than advancing first, precisely so an egg
  stays visible at the catch point for a full tick before its catch is decided.
  That preserves the player's ~1 s reaction window and makes the refactored
  human client indistinguishable from the original.
* **Dropped-egg fall speed (`dropped_advance`, default 2).** The original
  advanced falling eggs every 0.2 s — ~4–5× faster than the main cadence. Since
  a dropped egg has already cost its life and can no longer be interacted with,
  its fall is cosmetic; the only logical side effect is how long it keeps
  occupying a quadrant slot. `dropped_advance` re-expresses that "faster
  cadence" as integer ticks; it never affects catches, drops, score or lives.
* **Empty-quadrant vs. state-0 egg in the observation.** An empty quadrant and a
  just-spawned egg (`state 0`) both encode as `0`. Intentional: a fresh egg is
  maximally far from the catch point, so there's no actionable signal to lose.
* **Real-time pacing lives in the client, not the core.** The score-driven
  speed-up (`core.tick_seconds()`) is a pure function of score that the human
  client uses to time its ticks; the core itself is unitless and tick-based.