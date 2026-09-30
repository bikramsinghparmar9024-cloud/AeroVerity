"""Write viewer payloads into a run directory.

The viewer reads two files, written as the pipeline progresses:
  preview.json   the T0 preview (fast track), available within seconds
  accepted.json  the verified trajectory, sparse points and the verifier's reasons
Files are written atomically so the viewer never reads a half-written payload.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aerometric.outputs.report import json_safe
from aerometric.pose.classical import SparseModel
from aerometric.pose.feedforward import preview_payload
from aerometric.pose.types import Trajectory
from aerometric.pose.verifier import VerificationResult

PREVIEW_FILE = "preview.json"
ACCEPTED_FILE = "accepted.json"
MAX_VIEWER_POINTS = 30_000


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # strict JSON: NaN/Infinity become null, otherwise the browser cannot parse the file at all
    tmp.write_text(json.dumps(json_safe(payload), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_preview(run_dir: str | Path, traj: Trajectory, title: str = "") -> Path:
    payload = preview_payload(traj)
    payload["title"] = title
    path = Path(run_dir) / PREVIEW_FILE
    _write_json(path, payload)
    return path


def accepted_payload(result: VerificationResult, sparse: SparseModel, title: str = "",
                     max_points: int = MAX_VIEWER_POINTS, seed: int = 0) -> dict:
    payload = preview_payload(result.trajectory)
    pts, err = sparse.points, sparse.point_error_px
    if len(pts) > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(len(pts), max_points, replace=False))
        pts, err = pts[keep], err[keep]
    payload.update({
        "kind": "accepted",
        "title": title,
        "accepted": result.accepted,
        "reasons": list(result.reasons),
        "metrics": {"ai": asdict(result.ai), "classical": asdict(result.classical)},
        "points": np.round(pts, 3).tolist(),
        "point_error_px": np.round(err, 2).tolist(),
        "points_total": int(len(sparse.points)),
    })
    return payload


def write_accepted(run_dir: str | Path, result: VerificationResult, sparse: SparseModel,
                   title: str = "") -> Path:
    path = Path(run_dir) / ACCEPTED_FILE
    _write_json(path, accepted_payload(result, sparse, title))
    return path
