"""CLI entry point for recording a rollout to an explorable JSONL log.

Kept in its own module (not imported by ``nupogodi/__init__``) so that
``python -m nupogodi.record`` executes cleanly without runpy's
"found in sys.modules" double-import warning.

    python -m nupogodi.record                    # 20 episodes, then exit
    python -m nupogodi.record --episodes 500      # a longer run
    python -m nupogodi.record --steps 50000       # bound by steps instead
    python -m nupogodi.record --watch             # live view, run until Ctrl-C
    python -m nupogodi.record --out runs/my.jsonl # fixed output path

Logging goes through ``loguru``; ``--watch`` renders a live ``rich`` metrics
table so you can watch a run progress in the terminal (a lightweight preview of
the upcoming web dashboard). The default one-shot finishes in milliseconds at
~100k steps/sec, so it looks instant — that is expected.
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

from .agents.random_agent import RandomAgent
from .env import NuPogodiEnv
from .recorder import JsonlRecorder
from .rollout import EpisodeSummary, run

console = Console()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nupogodi-record",
        description="Record a random-agent rollout to a JSONL log in runs/.",
    )
    bound = p.add_mutually_exclusive_group()
    bound.add_argument(
        "--episodes", type=int, default=None, help="stop after N complete episodes"
    )
    bound.add_argument(
        "--steps", type=int, default=None, help="stop after N env steps"
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="live rich view; record continuously until interrupted (Ctrl-C)",
    )
    p.add_argument("--seed", type=int, default=0, help="rollout seed (default: 0)")
    p.add_argument(
        "--out", default=None, help="output path (default: runs/run-<timestamp>.jsonl)"
    )
    p.add_argument(
        "--max-bytes",
        type=_parse_size,
        default=1 << 30,
        help="roll to a new .jsonl part past this size, e.g. 1G / 512M (default: 1G)",
    )
    return p.parse_args(argv)


def _parse_size(text: str) -> int:
    """Parse a byte count with an optional K/M/G/T suffix (``512M``, ``1G``)."""
    s = text.strip().upper()
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    mult = units.get(s[-1:], 1)
    if s and s[-1] in units:
        s = s[:-1]
    return int(float(s) * mult)


class _LiveMetrics:
    """Sink that accumulates running stats and drives a rich live table."""

    def __init__(self, agent: str, refresh_hz: float = 8.0) -> None:
        self.agent = agent
        self.steps = 0
        self.episodes = 0
        self.best_score = 0
        self.last_score = 0
        self.last_reward = 0.0
        self._reward_window: deque[float] = deque(maxlen=100)
        self._start = time.perf_counter()
        self._min_interval = 1.0 / refresh_hz
        self._last_render = 0.0
        self.live: Live | None = None

    # -- Sink protocol -----------------------------------------------------

    def on_step(self, step_index: int, transition) -> None:
        self.steps += 1

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        self.episodes += 1
        self.best_score = max(self.best_score, summary.score)
        self.last_score = summary.score
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
    def steps_per_sec(self) -> float:
        return self.steps / self.elapsed if self.elapsed else 0.0

    @property
    def mean_reward(self) -> float:
        w = self._reward_window
        return sum(w) / len(w) if w else 0.0

    def render(self) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="cyan", no_wrap=True)
        table.add_column(justify="left")
        table.add_row("elapsed", f"{self.elapsed:6.1f} s")
        table.add_row("episodes", f"{self.episodes:,}")
        table.add_row("steps", f"{self.steps:,}")
        table.add_row("steps/sec", f"{self.steps_per_sec:,.0f}")
        table.add_row("best score", f"[bold green]{self.best_score}[/]")
        table.add_row("mean reward (last 100)", f"{self.mean_reward:+.2f}")
        table.add_row(
            "last episode", f"score {self.last_score}, reward {self.last_reward:+.0f}"
        )
        return Panel(
            table,
            title=f"nupogodi · recording ([magenta]{self.agent}[/] agent)",
            subtitle="Ctrl-C to stop",
            border_style="blue",
        )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env = NuPogodiEnv()
    agent = RandomAgent(seed=args.seed)
    meta = {"agent": "random", "seed": args.seed}

    if args.watch:
        _watch(env, agent, meta, path=args.out, max_bytes=args.max_bytes)
        return

    # Default one-shot: honor whichever bound is given, else 20 episodes.
    if args.steps is not None:
        bound = {"steps": args.steps}
    else:
        bound = {"episodes": args.episodes if args.episodes is not None else 20}

    with JsonlRecorder(path=args.out, meta=meta, max_bytes=args.max_bytes) as rec:
        stats = run(env, agent, seed=args.seed, sinks=[rec], **bound)
    logger.info(
        "recorded {} episodes / {} steps (best score {}) -> {}",
        stats.episodes,
        stats.steps,
        stats.best_score,
        rec.path,
    )
    console.print(_summary_table(stats, rec.path))


def _summary_table(stats, path) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="cyan")
    table.add_column(justify="left")
    table.add_row("episodes", f"{stats.episodes:,}")
    table.add_row("steps", f"{stats.steps:,}")
    table.add_row("steps/sec", f"{stats.steps_per_sec:,.0f}")
    table.add_row("best score", f"[bold green]{stats.best_score}[/]")
    table.add_row("mean episode reward", f"{stats.mean_episode_reward:+.2f}")
    table.add_row("log", str(path))
    return Panel(table, title="recording complete", border_style="green")


def _watch(
    env: NuPogodiEnv, agent: RandomAgent, meta: dict, *, path=None, max_bytes=1 << 30
) -> None:
    """Record until Ctrl-C, showing a live rich metrics table.

    A single ``run`` with an effectively-unbounded episode budget, so the
    recorder opens and closes exactly once no matter when the interrupt lands.
    """
    metrics = _LiveMetrics(agent=str(meta.get("agent", "?")))
    with JsonlRecorder(
        path=path, meta={**meta, "watch": True}, max_bytes=max_bytes
    ) as rec:
        logger.info("watching -> {}  (Ctrl-C to stop)", rec.path)
        with Live(metrics.render(), console=console, refresh_per_second=8) as live:
            metrics.live = live
            try:
                run(env, agent, episodes=10**18, sinks=[rec, metrics])
            except KeyboardInterrupt:
                pass
            finally:
                metrics.live = None
                live.update(metrics.render())
    logger.info(
        "stopped: {} episodes / {} steps (best score {}) -> {}",
        metrics.episodes,
        metrics.steps,
        metrics.best_score,
        rec.path,
    )


if __name__ == "__main__":
    main()
