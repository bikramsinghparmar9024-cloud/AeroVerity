"""Reconstruction jobs started from the browser.

Each upload becomes a run folder under the runs root:

    runs/<id>/input/video.<ext>, telemetry.<ext>[, intrinsics.json]
    runs/<id>/status.json      state, current stage, message
    runs/<id>/...              everything the pipeline writes (viewer files, products, report)

Jobs run one at a time on a worker thread (a reconstruction uses the whole
CPU). Status is written to disk so it survives a page reload or a restart.
"""

from __future__ import annotations

import json
import queue
import re
import secrets
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from aerometric.config import PipelineConfig
from aerometric.outputs.report import ReportBuilder

STAGES = ("Ingestion", "Synchronisation", "Pre-flight check", "Keyframes", "Matching", "Calibration",
          "Camera poses", "Coverage", "Georeferencing", "Depth", "Surface", "Uncertainty", "Trust",
          "Re-fly advice", "Outputs")
INPUT_TYPES = {
    "video": (".mp4", ".mov", ".avi", ".mkv", ".m4v"),
    "telemetry": (".srt", ".csv"),
    "intrinsics": (".json",),
}
RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{3,60}$")
DOWNLOADS = {  # published name -> path inside the run folder
    "model.glb": "products/model.glb", "model.ply": "products/model.ply", "points.las": "products/points.las",
    "dsm.tif": "products/dsm.tif", "ortho.tif": "products/ortho.tif", "splat_seed.ply": "products/splat_seed.ply",
    "trust.tif": "products/trust.tif", "classes.geojson": "products/classes.geojson",
    "refly_advice.json": "products/refly_advice.json", "quality_report.json": "report/quality_report.json",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:30] or "flight"
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug}-{secrets.token_hex(2)}"


@dataclass
class Job:
    id: str
    folder: Path

    @property
    def status_file(self) -> Path:
        return self.folder / "status.json"

    def read(self) -> dict:
        try:
            # utf-8-sig: tolerate a byte-order mark (Windows tools add one when editing by hand)
            return json.loads(self.status_file.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"id": self.id, "state": "unknown"}

    def write(self, **changes) -> dict:
        s = self.read()
        s.update(changes, id=self.id, updated=_now())
        tmp = self.status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(s), encoding="utf-8")
        tmp.replace(self.status_file)
        return s

    def input_file(self, kind: str) -> Path | None:
        return next(iter(sorted((self.folder / "input").glob(f"{kind}.*"))), None)


class StatusReport(ReportBuilder):
    """Report builder that also records progress for the web page."""

    def __init__(self, *args, job: Job, **kwargs):
        super().__init__(*args, **kwargs)
        self._job = job

    def add(self, stage, summary, *args, **kwargs):
        entry = super().add(stage, summary, *args, **kwargs)
        done = [e["stage"] for e in self.entries]
        nxt = next((s for s in STAGES if s not in done), None)
        self._job.write(state="running", stage=nxt or "Finishing", stages_done=done,
                        message=f"{stage}: {summary}")
        return entry


def run_pipeline(job: Job, cfg: PipelineConfig, key_path: Path | None) -> dict:
    """The real reconstruction; returns fields to store in the final status."""
    from aerometric.ingest.flight import FlightInputs
    from aerometric.pipeline.back import run_back
    from aerometric.pipeline.front import run_front

    s = job.read()
    title = s.get("title") or job.id
    rep = StatusReport(title, job.id, cfg.to_dict(), job=job)
    inputs = FlightInputs(job.input_file("video"), job.input_file("telemetry"),
                          intrinsics=job.input_file("intrinsics"))
    front = run_front(inputs, job.folder, cfg, rep, title, force=bool(s.get("force")))
    if front.stopped:
        problems = [c.message for c in front.gate.checks if c.status == "fail"]
        return {"state": "stopped", "message": "The pre-flight check does not expect a usable model: "
                + " ".join(problems)}
    back = run_back(front, job.folder, cfg, key_path=key_path, title=title)
    fr = back.trust.fractions(back.mesh.vertices, back.mesh.faces)
    return {"state": "done", "trust": fr,
            "message": f"{fr['measured']:.0%} of the surface is Measured, {fr['inferred']:.0%} Inferred, "
                       f"{fr['unknown']:.0%} Unknown."}


class JobQueue:
    def __init__(self, root: Path, cfg: PipelineConfig | None = None, key_path: Path | None = None,
                 runner: Callable[[Job, PipelineConfig, Path | None], dict] = run_pipeline) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or PipelineConfig()
        self.key_path = key_path
        self._runner = runner
        self._q: queue.Queue[Job] = queue.Queue()
        for s in self.list():  # runs cut off by a previous shutdown cannot resume
            if s.get("state") in ("queued", "running"):
                Job(s["id"], self.root / s["id"]).write(
                    state="failed", message="Interrupted: the app was closed while this run was in progress.")
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    # ---------------------------------------------------------------- run folders

    def get(self, run_id: str) -> Job | None:
        if not RUN_ID.match(run_id or ""):
            return None
        folder = self.root / run_id
        return Job(run_id, folder) if folder.is_dir() else None

    def create(self, title: str) -> Job:
        run_id = new_run_id(title)
        folder = self.root / run_id
        (folder / "input").mkdir(parents=True)
        job = Job(run_id, folder)
        job.write(state="waiting_for_files", title=title.strip() or "Untitled flight", created=_now(),
                  message="Waiting for the video and telemetry.")
        return job

    def list(self) -> list[dict]:
        out = []
        for d in self.root.iterdir():
            if d.is_dir() and RUN_ID.match(d.name) and (d / "status.json").is_file():
                out.append(Job(d.name, d).read())
        return sorted(out, key=lambda s: s.get("created", ""), reverse=True)

    # ---------------------------------------------------------------- execution

    def start(self, job: Job, force: bool = False) -> dict:
        missing = [k for k in ("video", "telemetry") if job.input_file(k) is None]
        if missing:
            raise ValueError(f"missing {' and '.join(missing)}")
        position = self._q.qsize() + (1 if self._busy() else 0)
        s = job.write(state="queued", force=force, stage=STAGES[0], stages_done=[],
                      message="Starting now." if position == 0 else f"Queued behind {position} other run(s).")
        self._q.put(job)
        return s

    def _busy(self) -> bool:
        return any(j.get("state") == "running" for j in self.list())

    def _work(self) -> None:
        while True:
            job = self._q.get()
            job.write(state="running", stage=STAGES[0], message="Reading the video and telemetry.")
            try:
                job.write(**self._runner(job, self.cfg, self.key_path), finished=_now())
            except Exception as exc:  # the page must learn about any failure, not just expected ones
                job.write(state="failed", message=str(exc) or exc.__class__.__name__, finished=_now(),
                          detail=traceback.format_exc(limit=5))
            finally:
                self._q.task_done()
