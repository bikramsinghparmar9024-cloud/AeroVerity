"""Local HTTP server for the viewer (standard library only, localhost by default).

Two modes:
  single run  `aerometric viewer RUN_DIR`   watch one run folder (read only)
  app         `aerometric app --runs DIR`   upload flights, queue them, browse every run

Routes (app mode adds the ones marked *):
  GET  /                                   the page
  GET  /api/config                         {"uploads": bool}
  GET  /api/state[?run=ID]                 preview / accepted payloads (+ job status in app mode)
* GET  /api/runs                           all runs, newest first
* POST /api/runs                           {"title": ...} -> new run folder, returns its status
* PUT  /api/runs/ID/files/KIND?name=FILE   raw file body; KIND = video | telemetry | intrinsics
* POST /api/runs/ID/start                  {"force": bool} -> queue the reconstruction
* GET  /runs/ID/report                     the run's quality report (HTML)
* GET  /runs/ID/download/NAME              a finished product (fixed list of names)
Nothing else on disk is reachable.
"""

from __future__ import annotations

import argparse
import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from aerometric.outputs.report import json_safe
from aerometric.viewer.export import ACCEPTED_FILE, PREVIEW_FILE

MAX_UPLOAD_BYTES = 16 * 1024 ** 3  # 16 GB: long 4K flights are large
CHUNK = 1 << 20


def read_state(run_dir: Path) -> dict:
    state: dict = {"stage": "waiting", "preview": None, "accepted": None}
    for key, name in (("preview", PREVIEW_FILE), ("accepted", ACCEPTED_FILE)):
        p = run_dir / name
        if p.is_file():
            try:
                state[key] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass  # being replaced right now; the next poll will pick it up
    if state["accepted"]:
        state["stage"] = "accepted"
    elif state["preview"]:
        state["stage"] = "preview"
    return state


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, run_dir: Path | None, jobs, **kwargs):
        self.run_dir, self.jobs = run_dir, jobs
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ routing

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        url = urlsplit(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        query = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", _page())  # read each time: cheap, and edits need no restart
        if parts == ["api", "config"]:
            return self._json({"uploads": self.jobs is not None})
        if parts == ["api", "state"]:
            return self._state(query.get("run", [None])[0])
        if self.jobs is None:
            return self._not_found()
        if parts == ["api", "runs"]:
            return self._json(self.jobs.list())
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "report":
            job = self.jobs.get(parts[1])
            page = job.folder / "report" / "quality_report.html" if job else None
            return self._file(page, "text/html; charset=utf-8") if page and page.is_file() else self._not_found()
        if len(parts) == 4 and parts[0] == "runs" and parts[2] == "download":
            from aerometric.viewer.jobs import DOWNLOADS

            job = self.jobs.get(parts[1])
            rel = DOWNLOADS.get(parts[3])
            f = job.folder / rel if job and rel else None
            if f is None or not f.is_file():
                return self._not_found()
            return self._file(f, "application/octet-stream", attachment=parts[3])
        return self._not_found()

    def do_POST(self) -> None:  # noqa: N802
        parts = [unquote(p) for p in urlsplit(self.path).path.strip("/").split("/") if p]
        if self.jobs is None:
            return self._not_found()
        if parts == ["api", "runs"]:
            body = self._read_json()
            title = str(body.get("title", "")).strip()[:120] if isinstance(body, dict) else ""
            return self._json(self.jobs.create(title).read(), 201)
        if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "start":
            job = self.jobs.get(parts[2])
            if job is None:
                return self._not_found()
            body = self._read_json()
            try:
                return self._json(self.jobs.start(job, force=bool(isinstance(body, dict) and body.get("force"))))
            except ValueError as exc:
                return self._json({"error": f"Cannot start yet: {exc}."}, 400)
        return self._not_found()

    def do_PUT(self) -> None:  # noqa: N802
        from aerometric.viewer.jobs import INPUT_TYPES

        url = urlsplit(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        if self.jobs is None or not (len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3] == "files"):
            return self._not_found()
        job, kind = self.jobs.get(parts[2]), parts[4]
        if job is None or kind not in INPUT_TYPES:
            return self._not_found()
        if job.read().get("state") not in ("waiting_for_files",):
            return self._json({"error": "This run has already started; create a new one to upload again."}, 409)
        name = parse_qs(url.query).get("name", [""])[0]
        ext = Path(name).suffix.lower()
        if ext not in INPUT_TYPES[kind]:
            return self._json({"error": f"A {kind} file must be one of: {', '.join(INPUT_TYPES[kind])}."}, 415)
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json({"error": "Upload is missing its size."}, 411)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            return self._json({"error": "File is empty or larger than 16 GB."}, 413)
        for old in (job.folder / "input").glob(f"{kind}.*"):
            old.unlink()
        target = job.folder / "input" / f"{kind}{ext}"
        partial_file = target.with_suffix(ext + ".part")
        remaining = length
        with partial_file.open("wb") as out:
            while remaining:
                chunk = self.rfile.read(min(CHUNK, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
        if remaining:
            partial_file.unlink(missing_ok=True)
            return self._json({"error": "The upload was interrupted; please try again."}, 400)
        partial_file.replace(target)
        job.write(**{f"{kind}_name": Path(name).name, f"{kind}_bytes": length})
        return self._json({"ok": True, "kind": kind, "bytes": length})

    # ------------------------------------------------------------------ helpers

    def _state(self, run_id: str | None) -> None:
        if self.jobs is None:
            return self._json(read_state(self.run_dir))
        job = self.jobs.get(run_id) if run_id else None
        if job is None:
            return self._json({"stage": "waiting", "preview": None, "accepted": None, "job": None})
        state = read_state(job.folder)
        state["job"] = job.read()
        return self._json(state)

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", "0") or 0)
            return json.loads(self.rfile.read(min(n, 65536)) or b"{}") if n else {}
        except (ValueError, json.JSONDecodeError):
            return {}

    def _json(self, obj, code: int = 200) -> None:
        # strict JSON for the browser, even for files written before NaN was sanitised
        body = json.dumps(json_safe(obj), allow_nan=False).encode("utf-8")
        self._send(code, "application/json", body)

    def _not_found(self) -> None:
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def _file(self, path: Path, ctype: str, attachment: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        with path.open("rb") as f:
            while chunk := f.read(CHUNK):
                self.wfile.write(chunk)

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # keep the console quiet
        pass


def _page() -> bytes:
    return resources.files("aerometric.viewer").joinpath("static/index.html").read_bytes()


def make_server(run_dir: str | Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Read-only viewer for one run folder."""
    return ThreadingHTTPServer((host, port), partial(_Handler, run_dir=Path(run_dir), jobs=None))


def make_app_server(jobs, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Upload + queue + browse, backed by a JobQueue."""
    return ThreadingHTTPServer((host, port), partial(_Handler, run_dir=None, jobs=jobs))


def serve_in_background(run_dir: str | Path, host: str = "127.0.0.1", port: int = 0
                        ) -> tuple[ThreadingHTTPServer, str]:
    server = make_server(run_dir, host, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{host}:{server.server_address[1]}/"


def main() -> None:
    ap = argparse.ArgumentParser(description="View a reconstruction run in the browser.")
    ap.add_argument("run_dir", help="run directory containing preview.json / accepted.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = make_server(args.run_dir, args.host, args.port)
    print(f"Viewer running at http://{args.host}:{server.server_address[1]}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
