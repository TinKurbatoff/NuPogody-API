"""Rollout driver — one place that runs an env+agent loop.

The loop that ties a :class:`~nupogodi.env.NuPogodiEnv` to an
:class:`~nupogodi.agents.base.Agent` used to be duplicated inline in every
caller. This module owns it once and fans every transition out to zero or more
*sinks* (a JSONL recorder, a metrics collector, later a live web push). The hot
training path pays nothing when ``sinks`` is empty, so the same driver serves
both a 100k-steps/sec headless benchmark and an instrumented, observable run.

Whatever trains the future spiking-network agent hooks in here too: give it an
:class:`Agent` and a list of sinks and it gets the same explorable telemetry.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from .agents.base import Agent, Transition
from .env import NuPogodiEnv


@dataclass
class EpisodeSummary:
    """Per-episode aggregate emitted to sinks when an episode ends."""

    episode: int
    steps: int
    total_reward: float
    score: int
    caught: int
    dropped: int
    spawned: int
    terminated: bool
    truncated: bool


@dataclass
class RolloutStats:
    """Whole-run aggregate returned by :func:`run`."""

    steps: int
    episodes: int
    steps_per_sec: float
    total_reward: float
    best_score: int
    mean_episode_reward: float


@runtime_checkable
class Sink(Protocol):
    """Consumes rollout events. Every method is optional in spirit; concrete
    sinks implement what they need. The driver calls all three."""

    def on_step(self, step_index: int, transition: Transition) -> None: ...

    def on_episode_end(self, summary: EpisodeSummary) -> None: ...

    def close(self) -> None: ...


def run(
    env: NuPogodiEnv,
    agent: Agent,
    *,
    steps: int | None = None,
    episodes: int | None = None,
    seed: int | None = None,
    sinks: list[Sink] | tuple[Sink, ...] = (),
) -> RolloutStats:
    """Run ``env`` under ``agent`` and stream transitions to ``sinks``.

    Bound the run by ``steps`` (total env steps, resetting on episode end) or by
    ``episodes`` (number of completed episodes); pass exactly one. Returns a
    :class:`RolloutStats` covering the whole run.
    """
    if (steps is None) == (episodes is None):
        raise ValueError("pass exactly one of `steps` or `episodes`")

    obs, _ = env.reset(seed=seed)

    step_index = 0
    episode_index = 0
    ep_steps = 0
    ep_reward = 0.0
    ep_caught = ep_dropped = ep_spawned = 0
    total_reward = 0.0
    best_score = 0
    episode_rewards: list[float] = []

    start = time.perf_counter()
    while True:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        transition = Transition(
            obs, action, reward, next_obs, terminated, truncated, info
        )
        agent.observe(transition)
        for sink in sinks:
            sink.on_step(step_index, transition)

        step_index += 1
        ep_steps += 1
        ep_reward += reward
        total_reward += reward
        ep_caught += info.get("caught", 0)
        ep_dropped += info.get("dropped", 0)
        ep_spawned += info.get("spawned", 0)
        obs = next_obs

        if terminated or truncated:
            summary = EpisodeSummary(
                episode=episode_index,
                steps=ep_steps,
                total_reward=ep_reward,
                score=int(info.get("score", 0)),
                caught=ep_caught,
                dropped=ep_dropped,
                spawned=ep_spawned,
                terminated=terminated,
                truncated=truncated,
            )
            for sink in sinks:
                sink.on_episode_end(summary)
            episode_rewards.append(ep_reward)
            best_score = max(best_score, summary.score)
            episode_index += 1
            ep_steps = 0
            ep_reward = 0.0
            ep_caught = ep_dropped = ep_spawned = 0

            if episodes is not None and episode_index >= episodes:
                break
            obs, _ = env.reset()

        if steps is not None and step_index >= steps:
            break

    elapsed = time.perf_counter() - start
    for sink in sinks:
        sink.close()

    return RolloutStats(
        steps=step_index,
        episodes=episode_index,
        steps_per_sec=step_index / elapsed if elapsed else float("inf"),
        total_reward=total_reward,
        best_score=best_score,
        mean_episode_reward=(
            sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        ),
    )


def _summary_dict(summary: EpisodeSummary) -> dict:
    return asdict(summary)
