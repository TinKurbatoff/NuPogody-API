"""CLI to train the spiking :class:`~nupogodi.agents.cl1_agent.CL1Agent`.

Trains through the very same ``rollout.run`` + ``Sink`` machinery the random
agent uses, so every transition streams to the JSONL recorder and the web
dashboard — you watch the culture learn *live*, mean reward climbing off the
~-1.9 random baseline.

    python -m nupogodi.train --episodes 400            # train, checkpoint, exit
    python -m nupogodi.train --watch                    # train until Ctrl-C, live view
    python -m nupogodi.train --episodes 400 --out models/cl1.pt
    # then, in another shell:  python -m nupogodi.dashboard

Kept out of ``nupogodi/__init__`` (like ``record.py``) so ``python -m
nupogodi.train`` runs without a runpy double-import warning.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .agents.cl1_agent import CL1Agent
from .cl1 import open as open_neurons
from .env import NuPogodiEnv
from .record import _parse_size
from .recorder import JsonlRecorder
from .rollout import EpisodeSummary, run

console = Console()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nupogodi-train",
        description="Train the CL1 spiking agent (reward-modulated STDP).",
    )
    bound = p.add_mutually_exclusive_group()
    bound.add_argument(
        "--episodes", type=int, default=None, help="stop after N episodes"
    )
    bound.add_argument("--steps", type=int, default=None, help="stop after N env steps")
    p.add_argument(
        "--watch", action="store_true", help="train until Ctrl-C, live view"
    )
    p.add_argument("--seed", type=int, default=0, help="seed (default: 0)")
    p.add_argument(
        "--out", default=None, help="checkpoint path (default: models/cl1-<ts>.pt)"
    )
    p.add_argument("--log", default=None, help="JSONL log path (default: runs/…jsonl)")
    p.add_argument(
        "--neurons", type=int, default=100, help="culture size (default: 100)"
    )
    p.add_argument(
        "--nu", type=float, default=0.05, help="STDP learning rate (default: 0.05)"
    )
    p.add_argument(
        "--gamma", type=float, default=0.9, help="TD discount (default: 0.9)"
    )
    p.add_argument(
        "--window", type=int, default=12, help="decision window ticks (default: 12)"
    )
    p.add_argument(
        "--max-bytes", type=_parse_size, default=1 << 30,
        help="roll JSONL to a new part past this size, e.g. 1G/512M (default: 1G)",
    )
    return p.parse_args(argv)


class _TrainMetrics:
    """Live-metrics sink: the recording panel plus learning-specific rows."""

    def __init__(self, agent: CL1Agent, refresh_hz: float = 8.0) -> None:
        self.agent = agent
        self.steps = 0
        self.episodes = 0
        self.best_score = 0
        self.last_reward = 0.0
        self._reward_window: deque[float] = deque(maxlen=100)
        self._delta_window: deque[float] = deque(maxlen=200)
        self._start = time.perf_counter()
        self._min_interval = 1.0 / refresh_hz
        self._last_render = 0.0
        self.live: Live | None = None

    # -- Sink protocol -----------------------------------------------------

    def on_step(self, step_index: int, transition) -> None:
        self.steps += 1
        self._delta_window.append(abs(self.agent.last_delta))

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        self.episodes += 1
        self.best_score = max(self.best_score, summary.score)
        self.last_reward = summary.total_reward
        self._reward_window.append(summary.total_reward)
        now = time.perf_counter()
        if self.live is not None and now - self._last_render >= self._min_interval:
            self._last_render = now
            self.live.update(self.render())

    def close(self) -> None:
        pass

    # -- rendering ---------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    @property
    def mean_reward(self) -> float:
        w = self._reward_window
        return sum(w) / len(w) if w else 0.0

    @property
    def mean_abs_delta(self) -> float:
        w = self._delta_window
        return sum(w) / len(w) if w else 0.0

    def _weight_norm(self) -> float:
        neurons = self.agent.neurons
        if hasattr(neurons, "plastic_weight_norm"):
            return neurons.plastic_weight_norm()
        return 0.0

    def render(self) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="cyan", no_wrap=True)
        table.add_column(justify="left")
        table.add_row("elapsed", f"{self.elapsed:6.1f} s")
        table.add_row("episodes", f"{self.episodes:,}")
        table.add_row("steps", f"{self.steps:,}")
        sps = self.steps / self.elapsed if self.elapsed else 0
        table.add_row("steps/sec", f"{sps:,.0f}")
        table.add_row("best score", f"[bold green]{self.best_score}[/]")
        mr = self.mean_reward
        colour = "green" if mr > -1.9 else "yellow"
        table.add_row("mean reward (100)", f"[{colour}]{mr:+.2f}[/]  (rand ≈ -1.9)")
        table.add_row("mean |δ| (dopamine)", f"{self.mean_abs_delta:.3f}")
        table.add_row("plastic weight norm", f"{self._weight_norm():.2f}")
        return Panel(
            table,
            title="nupogodi · training [magenta]CL1 spiking agent[/] (MSTDPET)",
            subtitle="Ctrl-C to stop" if self.live else "done",
            border_style="blue",
        )


def _build(args: argparse.Namespace) -> tuple[NuPogodiEnv, CL1Agent, dict]:
    env = NuPogodiEnv(flatten_obs=True)
    dish = open_neurons(backend="sim", neurons=args.neurons, nu=args.nu, seed=args.seed)
    agent = CL1Agent(dish, window=args.window, gamma=args.gamma, seed=args.seed)
    meta = {
        "agent": "cl1", "seed": args.seed, "neurons": args.neurons,
        "nu": args.nu, "gamma": args.gamma, "window": args.window,
    }
    return env, agent, meta


def _checkpoint_path(out: str | None) -> str:
    import pathlib

    if out is not None:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        return out
    models = pathlib.Path("models")
    models.mkdir(parents=True, exist_ok=True)
    return str(models / f"cl1-{time.strftime('%Y%m%d-%H%M%S')}.pt")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env, agent, meta = _build(args)
    ckpt = _checkpoint_path(args.out)

    if args.watch:
        _watch(env, agent, meta, args, ckpt)
        return

    bound = {"steps": args.steps} if args.steps is not None else {
        "episodes": args.episodes if args.episodes is not None else 200
    }
    metrics = _TrainMetrics(agent)
    with JsonlRecorder(path=args.log, meta=meta, max_bytes=args.max_bytes) as rec:
        logger.info("training CL1 agent -> log {}  ckpt {}", rec.path, ckpt)
        stats = run(env, agent, seed=args.seed, sinks=[rec, metrics], **bound)
    agent.save(ckpt)
    logger.info(
        "trained {} episodes / {} steps; mean reward {:+.2f}, best score {} -> {}",
        stats.episodes, stats.steps, stats.mean_episode_reward, stats.best_score, ckpt,
    )
    console.print(metrics.render())


def _watch(env, agent, meta, args, ckpt) -> None:
    metrics = _TrainMetrics(agent)
    watch_meta = {**meta, "watch": True}
    with JsonlRecorder(path=args.log, meta=watch_meta, max_bytes=args.max_bytes) as rec:
        logger.info("watching training -> {}  (Ctrl-C to stop)", rec.path)
        with Live(metrics.render(), console=console, refresh_per_second=8) as live:
            metrics.live = live
            try:
                run(env, agent, seed=args.seed, sinks=[rec, metrics], episodes=10**18)
            except KeyboardInterrupt:
                pass
            finally:
                metrics.live = None
                live.update(metrics.render())
    agent.save(ckpt)
    logger.info("stopped: {} episodes; checkpoint -> {}", metrics.episodes, ckpt)


if __name__ == "__main__":
    main()
