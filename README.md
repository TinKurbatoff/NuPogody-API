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

Четыре кнопки положения выбирают угол корзины напрямую — как на настоящем
четырёхкнопочном пульте:

* **A** — влево-вверх
* **Z** — влево-вниз
* **M** — вправо-вниз
* **K** — вправо-вверх
* **Стрелки** — перемещение по двум осям
* **1** — Включить/выключить звук
* **ENTER** — перезапуск

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
  rollout.py      # run(env, agent, *, steps|episodes, sinks): the one env↔agent loop.
  recorder.py     # JsonlRecorder sink -> explorable, tailable JSONL log in runs/.
  dashboard.py    # dependency-free live web UI that tails the newest run.
  clients/human.py    # keyboard -> action -> env.step -> render (the "human agent").
  agents/
    base.py           # the Agent protocol: act(obs) -> action, observe(transition).
    random_agent.py   # uniform-random baseline (also the throughput benchmark).
    cl1_agent.py      # CL1Agent: spiking policy driven through the CL neural interface.
  cl1/                # the CL1-shaped neural interface (one contract, two backends)
    api.py            #   ChannelSet/StimDesign/BurstDesign + NeuronsLike protocol.
    dish.py           #   SpikingDish: responsive BindsNET culture that learns (MSTDPET).
    hardware.py       #   RealDish: thin adapter over the real Cortical Labs `cl` SDK.
  record.py / train.py    # CLIs: record a random rollout / train the spiking agent.
  transport/ws_server.py  # WebSocket wrapper (stub) for remote/human/bio clients.
tests/              # core, determinism, env-checker, renderer, agents, cl1.
```

**Three ways to drive the identical game core** — all speak the same
`env.step(action)` contract, so none of them touches the core:

| Variant | Who decides the action | Run it |
| --- | --- | --- |
| **Human** | you, on the keyboard | `nupogodi-human` |
| **Random** | a uniform baseline (also the speed benchmark) | `nupogodi-random` |
| **Spiking / CL1** | a spiking net learning by reward-modulated STDP, through the Cortical Labs interface | `nupogodi-train` |

## Install

```bash
pip install -e '.[dev]'      # core + tests + ruff + pygame
pip install -e '.[render]'   # add pygame only (for the human client)
pip install -e '.[snn]'      # add the spiking substrate (BindsNET) for the CL1 agent
pip install -e '.[cl]'       # optional: the real Cortical Labs cl-sdk (CL1 / official sim)
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

**(a) Human client** — play with the keyboard. Four position buttons pick a
basket quadrant directly, matching the real four-button device: `A` = left-up,
`Z` = left-down, `M` = right-down, `K` = right-up (arrow keys still move on the
two axes). `1` toggles sound, `SPACE` restarts, `ESC`/`Q` quits:

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

## (c) Spiking CL1 agent — playing through a Cortical Labs interface

The endgame agent is a **spiking neural network trained by reward-modulated
STDP**, driven through the *actual* Cortical Labs CL SDK interface shape so the
integration is rehearsed against the real hardware surface, not a bespoke one.

```bash
pip install -e '.[snn]'                 # BindsNET substrate (+ a torch._six shim)
python -m nupogodi.train --episodes 500 # train; or: nupogodi-train
python -m nupogodi.dashboard            # watch it learn live in the browser
pip install -e '.[cl]'                  # optional: real cl-sdk (CL1 / official sim)
```

**One contract, two backends.** The agent (`nupogodi/agents/cl1_agent.py`) speaks
only the CL-shaped `NeuronsLike` interface (`nupogodi/cl1/api.py`): rate-encode
the observation into electrode **stimulation** (`ChannelSet` / `StimDesign` /
`BurstDesign`, within the real ≤200 Hz-per-channel, 20 µs-quantum limits), run a
decision **window**, read spikes off the motor electrodes, take the most-active
action. Exploration comes from the culture's own intrinsic noise — no ε-greedy,
just like DishBrain. `nupogodi.cl1.open(backend=…)` hands the agent either:

* `"sim"` — a **responsive, learning** culture (`SpikingDish`, BindsNET
  `MSTDPET` = three-factor STDP with eligibility traces). Learning is delivered
  as a neuromodulatory signal δ (a critic's reward-prediction error = the
  dopamine analogue) — **no backprop**, because a CL1 can't do backprop.
* `"hardware"` — a thin adapter over real `cl` (`RealDish`), for CL1 or the
  official simulator. The identical agent code runs on both.

> **Status — honest.** The full closed loop works and the culture demonstrably
> learns in isolation (a fixed observation→action mapping shifts under reward).
> **Convergence to a strong *game* policy is an open research problem**: under
> the game's sparse, delayed reward the current global-δ readout still sits near
> the random-agent floor (mean episode reward ≈ −1.9). This is the known
> "R-STDP is hard to converge" trade-off of choosing biological fidelity over
> backprop. Tuning levers live on `SpikingDish`/`CL1Agent` (`tc_e_trace`, `nu`,
> `reward_scale`, `window`) and reward shaping is the likely next lever.

## Tests & performance

```bash
pytest          # core rules, determinism, gymnasium env-checker, renderer, agents
ruff check .
```

The spiking agent's tests skip cleanly when the `snn` extra isn't installed.

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