"""Tests for the shared rollout driver and the JSONL transition recorder."""

from __future__ import annotations

import json

import pytest

from nupogodi.agents.random_agent import RandomAgent
from nupogodi.env import NuPogodiEnv
from nupogodi.recorder import JsonlRecorder
from nupogodi.rollout import EpisodeSummary, RolloutStats, Sink, run


class _CountingSink:
    """Minimal sink that tallies the driver's callbacks."""

    def __init__(self) -> None:
        self.steps = 0
        self.episodes = 0
        self.closed = False
        self.last_step_index = -1

    def on_step(self, step_index, transition) -> None:
        self.steps += 1
        self.last_step_index = step_index

    def on_episode_end(self, summary) -> None:
        assert isinstance(summary, EpisodeSummary)
        self.episodes += 1

    def close(self) -> None:
        self.closed = True


def _run(**kwargs):
    return run(NuPogodiEnv(), RandomAgent(seed=0), seed=0, **kwargs)


def test_counting_sink_satisfies_protocol() -> None:
    assert isinstance(_CountingSink(), Sink)


def test_run_requires_exactly_one_bound() -> None:
    with pytest.raises(ValueError):
        _run()
    with pytest.raises(ValueError):
        _run(steps=10, episodes=1)


def test_run_by_steps_streams_every_step_to_sinks() -> None:
    sink = _CountingSink()
    stats = _run(steps=500, sinks=[sink])
    assert isinstance(stats, RolloutStats)
    assert stats.steps == 500
    assert sink.steps == 500
    assert sink.last_step_index == 499
    assert sink.closed
    assert sink.episodes == stats.episodes


def test_run_by_episodes_stops_after_n_episodes() -> None:
    sink = _CountingSink()
    stats = _run(episodes=5, sinks=[sink])
    assert stats.episodes == 5
    assert sink.episodes == 5
    assert stats.steps == sink.steps


def test_first_episode_is_reproducible_from_seed() -> None:
    # Only the first episode is seeded; the driver reseeds later episodes with
    # fresh entropy (mirrors the original loop), so bound to one episode here.
    a = _run(episodes=1)
    b = _run(episodes=1)
    assert (a.steps, a.total_reward, a.best_score) == (
        b.steps,
        b.total_reward,
        b.best_score,
    )


def test_recorder_writes_meta_step_and_episode_records(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    with JsonlRecorder(path=path, meta={"agent": "random", "seed": 0}) as rec:
        stats = run(
            NuPogodiEnv(), RandomAgent(seed=0), episodes=3, seed=0, sinks=[rec]
        )

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    kinds = [rec_["type"] for rec_ in lines]

    assert kinds[0] == "meta"
    assert lines[0]["agent"] == "random"
    assert kinds.count("episode") == 3 == stats.episodes
    assert kinds.count("step") == stats.steps

    step = next(rec_ for rec_ in lines if rec_["type"] == "step")
    # Full Gym tuple plus decoded game fields are present and JSON-clean.
    for key in ("obs", "action", "reward", "next_obs", "terminated", "truncated"):
        assert key in step
    assert step["action"] in {0, 1, 2, 3}
    assert isinstance(step["eggs"], list)


def test_record_cli_writes_a_log(tmp_path) -> None:
    from nupogodi import record

    out = tmp_path / "cli.jsonl"
    record.main(["--episodes", "3", "--out", str(out), "--seed", "0"])

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[0]["type"] == "meta"
    assert sum(1 for rec_ in lines if rec_["type"] == "episode") == 3


def test_recorder_flushes_incrementally_for_live_tailing(tmp_path) -> None:
    """A live viewer must see steps mid-run, not only after close()."""
    path = tmp_path / "live.jsonl"
    rec = JsonlRecorder(path=path, flush_each=True)
    env = NuPogodiEnv()
    agent = RandomAgent(seed=0)
    obs, _ = env.reset(seed=0)
    from nupogodi.agents.base import Transition

    for i in range(5):
        action = agent.act(obs)
        nxt, r, term, trunc, info = env.step(action)
        rec.on_step(i, Transition(obs, action, r, nxt, term, trunc, info))
        obs = nxt
        if term or trunc:
            break

    # File is readable and complete before close() is ever called.
    written = path.read_text().splitlines()
    assert sum(1 for line in written if '"type":"step"' in line) >= 1
    rec.close()


def test_recorder_rotates_into_parts_past_max_bytes(tmp_path) -> None:
    """A tiny cap forces several parts; every step survives across the roll."""
    path = tmp_path / "run.jsonl"
    # ~400 bytes/step, so a 1 KiB cap rolls every few steps.
    with JsonlRecorder(path=path, meta={"agent": "random"}, max_bytes=1024) as rec:
        stats = run(
            NuPogodiEnv(), RandomAgent(seed=0), steps=200, seed=0, sinks=[rec]
        )

    parts = sorted(tmp_path.glob("run*.jsonl"), key=lambda p: p.stat().st_mtime_ns)
    assert parts[0] == path  # part 0 keeps the exact base name.
    assert len(parts) > 1  # the cap actually forced a rotation.

    # Each part is self-describing: it leads with its own indexed meta line.
    for i, part in enumerate(parts):
        first = json.loads(part.read_text().splitlines()[0])
        assert first["type"] == "meta" and first["part"] == i

    # No step is lost or duplicated across the split — every part contributes,
    # and the total matches the driver's own count.
    steps = [
        json.loads(line)
        for part in parts
        for line in part.read_text().splitlines()
        if '"type":"step"' in line
    ]
    assert len(steps) == stats.steps
    assert [s["i"] for s in steps] == list(range(stats.steps))  # continuous index.


def test_max_bytes_zero_disables_rotation(tmp_path) -> None:
    """A falsy cap means one file, however big — the pre-rotation behaviour."""
    path = tmp_path / "run.jsonl"
    with JsonlRecorder(path=path, max_bytes=0) as rec:
        run(NuPogodiEnv(), RandomAgent(seed=0), steps=100, seed=0, sinks=[rec])
    assert list(tmp_path.glob("run*.jsonl")) == [path]
