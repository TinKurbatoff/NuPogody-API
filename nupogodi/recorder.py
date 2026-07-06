"""JSONL transition recorder — the explorable raw Gym-protocol log.

One JSON object per line, so a run is greppable, tailable, and loads straight
into ``pandas`` (``pd.read_json(path, lines=True)``). Three record ``type``s:

* ``meta``    — one header line: timestamp plus any caller-supplied config.
* ``step``    — the full Gym tuple for one env step (obs, action, reward,
  next_obs, terminated, truncated) plus decoded game fields (score, lives,
  per-tick caught/dropped/spawned counts, wolf quadrant, and each egg's
  ``[quadrant, state, dropped]``). This is the raw data to drill into.
* ``episode`` — a per-episode summary as the episode ends.

``flush_each=True`` (the default) flushes every line so a live viewer tailing
the file sees steps as they happen — which is what the upcoming web dashboard
polls. Turn it off for a fast post-hoc dump.

**Rotation.** A long run would otherwise grow one file without bound, so past
``max_bytes`` (default 1 GiB) the recorder rolls to the next *part*: part 0 keeps
the base name (``run-….jsonl``), part *k* becomes ``run-….00k.jsonl``. Every
part starts with its own ``meta`` line (carrying a ``part`` index) so each file
is self-describing and loads on its own. The parts are ordinary siblings in
``runs/``, so :mod:`nupogodi.dashboard` — which follows the newest ``*.jsonl`` —
switches to each new part automatically as it appears.

A recorder is a :class:`~nupogodi.rollout.Sink`; hand it to
:func:`nupogodi.rollout.run` via ``sinks=[recorder]``.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import numpy as np

from .agents.base import Transition
from .rollout import EpisodeSummary, _summary_dict
from .types import GameState

DEFAULT_RUN_DIR = pathlib.Path("runs")
DEFAULT_MAX_BYTES = 1 << 30  # 1 GiB — roll to a new part file past this size.


def _round(x: float) -> float:
    """Keep the log compact; obs floats past 4 dp are never meaningful here."""
    return round(float(x), 4)


def _to_list(obs: Any) -> list:
    arr = np.asarray(obs)
    if np.issubdtype(arr.dtype, np.floating):
        return [_round(v) for v in arr.tolist()]
    return [int(v) for v in arr.tolist()]


class JsonlRecorder:
    """Writes rollout events to a newline-delimited JSON file."""

    def __init__(
        self,
        path: str | pathlib.Path | None = None,
        *,
        run_dir: str | pathlib.Path = DEFAULT_RUN_DIR,
        meta: dict[str, Any] | None = None,
        flush_each: bool = True,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if path is None:
            run_dir = pathlib.Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self.path = pathlib.Path(path)  # part-0 / base name; also the run's id.
        self.flush_each = flush_each
        self.max_bytes = int(max_bytes) if max_bytes else 0  # 0/None disables.
        self._meta = dict(meta or {})
        self._i = 0  # own monotonic step counter, continuous across parts.
        self._part = 0
        self._bytes = 0
        self._open_part()

    # -- Sink protocol -----------------------------------------------------

    def on_step(self, step_index: int, transition: Transition) -> None:
        self._maybe_rotate()
        info = transition.info
        record: dict[str, Any] = {
            "type": "step",
            "i": self._i,
            "obs": _to_list(transition.obs),
            "action": int(transition.action),
            "reward": _round(transition.reward),
            "next_obs": _to_list(transition.next_obs),
            "terminated": bool(transition.terminated),
            "truncated": bool(transition.truncated),
            "score": info.get("score"),
            "lives": info.get("lives"),
            "caught": info.get("caught", 0),
            "dropped": info.get("dropped", 0),
            "spawned": info.get("spawned", 0),
        }
        state = info.get("state")
        if isinstance(state, GameState):
            record["tick"] = state.tick
            record["wolf"] = int(state.wolf_quadrant)
            record["eggs"] = [
                [int(e.quadrant), e.state, e.dropped] for e in state.eggs
            ]
        self._write(record)
        self._i += 1

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        self._maybe_rotate()
        self._write({"type": "episode", **_summary_dict(summary)})

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    # -- internals ---------------------------------------------------------

    def _part_path(self, part: int) -> pathlib.Path:
        """Filename for part ``part``: the base name for 0, ``.NNN`` before the
        extension after that — ``run-….jsonl`` → ``run-….001.jsonl``."""
        if part == 0:
            return self.path
        suffix = self.path.suffix  # ".jsonl", kept so parts still match *.jsonl.
        stem = self.path.name[: -len(suffix)] if suffix else self.path.name
        return self.path.with_name(f"{stem}.{part:03d}{suffix}")

    def _open_part(self) -> None:
        """Open the current part and lead it with its own ``meta`` line."""
        self.current_path = self._part_path(self._part)
        self._fh = self.current_path.open("w", encoding="utf-8")
        self._bytes = 0
        self._write(
            {"type": "meta", "ts": time.time(), "part": self._part, **self._meta}
        )

    def _maybe_rotate(self) -> None:
        """Roll to the next part once the active file passes ``max_bytes``.

        Checked only between records (never mid-record), so a part slightly
        overshoots the cap by one line rather than splitting a JSON object.
        """
        if self.max_bytes and self._bytes >= self.max_bytes:
            self._fh.flush()
            self._fh.close()
            self._part += 1
            self._open_part()

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        self._fh.write(line)
        self._bytes += len(line.encode("utf-8"))
        if self.flush_each:
            self._fh.flush()

    def __enter__(self) -> JsonlRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
