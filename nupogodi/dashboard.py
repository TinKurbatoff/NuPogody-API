"""Live web dashboard — watch a recorded run in the browser.

A dependency-free ``http.server`` (stdlib only) that tails a JSONL log written
by :mod:`nupogodi.recorder` and serves a single self-contained HTML page. The
page polls a ``/tail`` endpoint ~1x/second and appends new records, so a run in
progress (``python -m nupogodi.record --watch``) shows up live: the episode
reward curve, running metrics, and a raw last-steps table for drilling in.

    # terminal 1 — produce a growing log
    python -m nupogodi.record --watch

    # terminal 2 — serve it
    python -m nupogodi.dashboard            # opens http://127.0.0.1:8770

By default it follows the newest file in ``runs/`` and re-resolves on every poll,
so when a new ``--watch`` run starts — or a long run rotates to its next part
(``run-….001.jsonl``) — the page switches to it automatically. Pin a specific
file with ``--log PATH``.

Everything is stdlib on purpose: no server-side chart/websocket deps, and the
page inlines its own CSS/JS (a hand-drawn canvas chart) so it needs no network.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from loguru import logger

DEFAULT_RUNS_DIR = pathlib.Path("runs")
DEFAULT_PORT = 8770
# Cap the bytes served per /tail response so a huge backlog is drained across
# several polls instead of one giant reply (which stalls the browser).
MAX_TAIL_BYTES = 8 * 1024 * 1024
# When first attaching to an already-large log, start this many bytes from the
# end instead of replaying gigabytes of history — a live "tail" view.
INITIAL_WINDOW_BYTES = 8 * 1024 * 1024
_PAGE = (pathlib.Path(__file__).parent / "dashboard.html").read_text(
    encoding="utf-8"
)


def latest_log(runs_dir: pathlib.Path) -> pathlib.Path | None:
    """Newest ``*.jsonl`` in ``runs_dir``, or None if there are none.

    Sorted by nanosecond mtime so that when the recorder rotates to a new part
    (see :mod:`nupogodi.recorder`) the fresh part wins even if it lands in the
    same wall-clock second — the dashboard then follows it automatically.
    """
    logs = sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime_ns)
    return logs[-1] if logs else None


def initial_offset(path: pathlib.Path, window: int = INITIAL_WINDOW_BYTES) -> int:
    """Byte offset to start a fresh client at: the tail ``window`` of the file.

    For a small log this is 0 (replay everything). For an already-large log it
    jumps near the end — aligned to the next line boundary — so the browser gets
    a bounded, immediately-visible live view instead of gigabytes of history.
    """
    size = path.stat().st_size
    if size <= window:
        return 0
    start = size - window
    with path.open("rb") as fh:
        fh.seek(start)
        chunk = fh.read(1 << 16)
    nl = chunk.find(b"\n")
    return start + nl + 1 if nl != -1 else start


def read_records_since(
    path: pathlib.Path, offset: int, max_bytes: int = MAX_TAIL_BYTES
) -> tuple[list[dict], int]:
    """Return JSON records after byte ``offset`` plus the new byte offset.

    Reads only complete lines (stops at the last newline) so a partially-written
    final line from a live writer is not parsed until it is finished. At most
    ``max_bytes`` are read per call, so a large backlog is drained over several
    polls rather than one huge reply. If the file shrank since ``offset``
    (rotated/replaced) it restarts from the beginning.
    """
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        if offset > size:
            offset = 0
        fh.seek(offset)
        data = fh.read(max_bytes)

    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], offset
    complete = data[: last_nl + 1]
    new_offset = offset + last_nl + 1
    records = []
    for line in complete.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip a corrupt line rather than kill the stream
    return records, new_offset


class _Handler(BaseHTTPRequestHandler):
    # Set per-server in serve(); the handler is instantiated per request.
    runs_dir: pathlib.Path = DEFAULT_RUNS_DIR
    pinned: pathlib.Path | None = None

    def log_message(self, *args) -> None:  # silence default stderr spam.
        pass

    def _target(self) -> pathlib.Path | None:
        return self.pinned if self.pinned is not None else latest_log(self.runs_dir)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path == "/":
            self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
        elif route.path == "/tail":
            self._tail(parse_qs(route.query))
        elif route.path == "/favicon.ico":
            self._send(204, "text/plain", b"")
        else:
            self._send(404, "text/plain", b"not found")

    def _tail(self, query: dict[str, list[str]]) -> None:
        target = self._target()
        if target is None:
            self._json({"file": None, "records": [], "offset": 0, "reset": True})
            return
        client_file = query.get("file", [""])[0]
        offset = int(query.get("offset", ["0"])[0])
        reset = client_file != target.name
        if reset:
            offset = initial_offset(target)
        records, new_offset = read_records_since(target, offset)
        self._json(
            {
                "file": target.name,
                "records": records,
                "offset": new_offset,
                "reset": reset,
            }
        )

    def _json(self, payload: dict) -> None:
        self._send(200, "application/json", json.dumps(payload).encode("utf-8"))

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    runs_dir: pathlib.Path = DEFAULT_RUNS_DIR,
    log: pathlib.Path | None = None,
    open_browser: bool = True,
) -> None:
    _Handler.runs_dir = runs_dir
    _Handler.pinned = log
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    target = log if log is not None else latest_log(runs_dir)
    logger.info("dashboard -> {}  (following {})", url, target or f"{runs_dir}/*.jsonl")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("dashboard stopped")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="nupogodi-dashboard",
        description="Serve a live browser view of a recorded run.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--runs-dir", type=pathlib.Path, default=DEFAULT_RUNS_DIR,
        help="directory of run logs to follow (default: runs/)",
    )
    p.add_argument(
        "--log", type=pathlib.Path, default=None,
        help="pin one log file instead of following the newest",
    )
    p.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = p.parse_args(argv)
    serve(
        host=args.host,
        port=args.port,
        runs_dir=args.runs_dir,
        log=args.log,
        open_browser=not args.no_open,
    )



if __name__ == "__main__":
    main()
