import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from aerometric.pose.classical import SparseModel
from aerometric.pose.types import CameraPose, Trajectory
from aerometric.pose.verifier import TrajectoryMetrics, VerificationResult
from aerometric.types import CameraIntrinsics
from aerometric.viewer.export import accepted_payload, write_accepted, write_preview
from aerometric.viewer.server import serve_in_background

K = CameraIntrinsics(1920, 1080, 1400, 1400, 960, 540, source="provided")


def traj(source="mock-feedforward", is_mock=True):
    poses = tuple(CameraPose.from_center(np.diag([1.0, -1.0, -1.0]), np.array([i * 10.0, 0, 100]))
                  for i in range(5))
    return Trajectory(np.arange(5) * 6, np.arange(5) * 0.2, poses, np.array([1, 1, 0, 1, 1], bool), K,
                      "enu", True, source, is_mock, ("stand-in note",))


def verification():
    m = TrajectoryMetrics(0.8, 1.9, 1.1, 1.2, 1.0, 900)
    return VerificationResult("classical", traj("scipy-ba", False), m, m, ("Using the classical track.",))


def sparse(n=50):
    rng = np.random.default_rng(0)
    return SparseModel(rng.normal(size=(n, 3)), tuple(np.zeros((2, 2), int) for _ in range(n)), rng.random(n))


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


@pytest.fixture
def server(tmp_path):
    srv, url = serve_in_background(tmp_path)
    yield tmp_path, url
    srv.shutdown()


def test_accepted_payload_subsamples_points():
    p = accepted_payload(verification(), sparse(100), "Test", max_points=30)
    assert len(p["points"]) == 30 and p["points_total"] == 100
    assert p["accepted"] == "classical" and p["is_mock"] is False
    assert p["metrics"]["ai"]["reproj_p90_px"] == pytest.approx(1.9)
    json.dumps(p)


def test_server_stages(server):
    run, url = server
    status, ctype, body = get(url)
    assert status == 200 and ctype.startswith("text/html") and b"Reconstruction Viewer" in body

    state = json.loads(get(url + "api/state")[2])
    assert state["stage"] == "waiting"

    write_preview(run, traj(), "Demo flight")
    state = json.loads(get(url + "api/state")[2])
    assert state["stage"] == "preview" and state["preview"]["is_mock"] is True
    assert state["preview"]["cameras"][2]["valid"] is False

    write_accepted(run, verification(), sparse(), "Demo flight")
    state = json.loads(get(url + "api/state")[2])
    assert state["stage"] == "accepted" and len(state["accepted"]["points"]) == 50


def test_nan_never_reaches_the_browser(server):
    run, url = server
    m = TrajectoryMetrics(0.8, float("nan"), float("inf"), 1.2, 1.0, 900)
    write_accepted(run, VerificationResult("classical", traj("scipy-ba", False), m, m, ("x",)), sparse(), "t")
    text = (run / "accepted.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    # an older file that still contains NaN is cleaned by the server too
    (run / "accepted.json").write_text(text.replace('"reproj_p90_px":null', '"reproj_p90_px":NaN'))
    raw = get(url + "api/state")[2].decode()
    assert "NaN" not in raw and json.loads(raw)["accepted"]["metrics"]["ai"]["reproj_p90_px"] is None


def test_server_serves_nothing_else(server):
    run, url = server
    (run / "secret.txt").write_text("x")
    for path in ("secret.txt", "../pyproject.toml", "preview.json", "static/index.html"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(url + path)
        assert exc.value.code == 404
