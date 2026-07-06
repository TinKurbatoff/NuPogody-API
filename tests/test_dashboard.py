"""Tests for the live dashboard's log-tailing and latest-file resolution."""

from __future__ import annotations

import time

from nupogodi.dashboard import initial_offset, latest_log, read_records_since


def test_read_records_since_reads_incrementally(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"type":"meta"}\n{"type":"step","i":0}\n')

    recs, offset = read_records_since(path, 0)
    assert [r["type"] for r in recs] == ["meta", "step"]
    assert offset == path.stat().st_size

    # Nothing new yet.
    recs, offset2 = read_records_since(path, offset)
    assert recs == []
    assert offset2 == offset

    # Append a line; only the new record comes back.
    with path.open("a") as fh:
        fh.write('{"type":"episode","episode":0}\n')
    recs, offset3 = read_records_since(path, offset)
    assert [r["type"] for r in recs] == ["episode"]
    assert offset3 == path.stat().st_size


def test_read_records_since_ignores_partial_last_line(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    # A half-written final line (no trailing newline) must not be parsed.
    path.write_text('{"type":"step","i":0}\n{"type":"step","i":1')
    recs, offset = read_records_since(path, 0)
    assert [r["i"] for r in recs] == [0]
    # Offset stops at the last newline, so the partial line is retried later.
    assert offset == len('{"type":"step","i":0}\n')


def test_read_records_since_caps_bytes_per_call(tmp_path) -> None:
    # A large backlog must be drained across polls, not in one giant reply.
    path = tmp_path / "run.jsonl"
    line = '{"type":"step","i":0}\n'
    with path.open("w") as fh:
        for _ in range(1000):
            fh.write(line)

    recs, offset = read_records_since(path, 0, max_bytes=len(line) * 10)
    assert len(recs) == 10  # only the first budget's worth
    assert offset == len(line) * 10
    # The rest is picked up on subsequent calls from the returned offset.
    recs2, _ = read_records_since(path, offset, max_bytes=len(line) * 10)
    assert len(recs2) == 10


def test_initial_offset_tails_a_large_file(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    with path.open("w") as fh:
        for i in range(500):
            fh.write(f'{{"type":"step","i":{i}}}\n')

    # Small file: replay from the start.
    assert initial_offset(path, window=10**9) == 0

    # Large file (tiny window): start near the end, aligned to a line boundary,
    # and every record read from there parses cleanly (no split line).
    off = initial_offset(path, window=200)
    assert off > 0
    recs, _ = read_records_since(path, off)
    assert recs and all(r["type"] == "step" for r in recs)
    assert recs[-1]["i"] == 499


def test_latest_log_picks_newest(tmp_path) -> None:
    assert latest_log(tmp_path) is None
    old = tmp_path / "a.jsonl"
    new = tmp_path / "b.jsonl"
    old.write_text("{}\n")
    time.sleep(0.01)
    new.write_text("{}\n")
    assert latest_log(tmp_path) == new
