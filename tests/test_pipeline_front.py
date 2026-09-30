import json

import numpy as np
import pytest

from aerometric.config import PipelineConfig
from aerometric.ingest.flight import FlightInputs
from aerometric.outputs.report import generate_key, verify_report
from aerometric.pipeline.front import run_front
from synthetic_flight import FRAME, make_flight


@pytest.fixture(scope="module")
def front(tmp_path_factory):
    folder = tmp_path_factory.mktemp("flight")
    fl = make_flight(folder)
    res = run_front(FlightInputs(fl.video, fl.srt, intrinsics=fl.intrinsics), folder / "run", PipelineConfig(),
                    title="Synthetic flight")
    return fl, res, folder


def test_front_half_runs_end_to_end(front):
    fl, res, folder = front
    assert not res.stopped
    assert res.offset.reliable and res.offset.offset_s == pytest.approx(fl.offset_s, abs=0.1)
    assert len(res.keyframes.keyframes) >= 6
    assert len(res.pairs) >= len(res.keyframes.keyframes) - 1
    assert res.verification.trajectory.world == "enu" and res.verification.trajectory.metric
    assert len(res.reconstruction.sparse.points) > 500


def test_cameras_land_where_the_drone_flew(front):
    fl, res, _ = front
    traj = res.verification.trajectory
    lat, lon, h = res.track.frame.to_lla(traj.centers)
    est = FRAME.to_enu(lat, lon, h)
    truth = fl.centers_at(np.array([k.t_video_s for k in res.keyframes.keyframes]))
    err = np.linalg.norm(est - truth, axis=1)
    assert np.median(err) < 1.5


def test_viewer_files_and_signed_report(front, tmp_path):
    fl, res, folder = front
    run = folder / "run"
    assert json.loads((run / "preview.json").read_text())["kind"] == "t0_preview"
    acc = json.loads((run / "accepted.json").read_text())
    assert acc["kind"] == "accepted" and acc["points_total"] > 500
    stages = [e["stage"] for e in res.report.entries]
    assert stages == ["Ingestion", "Synchronisation", "Pre-flight check", "Keyframes", "Matching", "Calibration",
                      "Camera poses", "Coverage"]
    doc = res.report.finalize(generate_key(tmp_path / "k.pem"))
    assert verify_report(doc, files_root=run).ok
