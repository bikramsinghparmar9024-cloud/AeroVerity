import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from aerometric.viewer.jobs import JobQueue, StatusReport
from aerometric.viewer.server import make_app_server


def fake_pipeline(job, cfg, key_path):
    """Stands in for a real reconstruction: walks the stages and writes a result file."""
    rep = StatusReport("t", job.id, {}, job=job)
    for stage in ("Ingestion", "Synchronisation"):
        rep.add(stage, "ok")
    (job.folder / "products").mkdir()
    (job.folder / "products" / "model.glb").write_bytes(b"glb!")
    if "fail" in job.read()["title"]:
        raise RuntimeError("telemetry has no GPS")
    return {"state": "done", "message": "all good"}


@pytest.fixture
def app(tmp_path):
    jobs = JobQueue(tmp_path / "runs", runner=fake_pipeline)
    srv = make_app_server(jobs, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/", tmp_path / "runs"
    srv.shutdown()


def call(url, method="GET", body=None, data=None):
    req = urllib.request.Request(url, method=method, data=data if data is not None else
                                 (json.dumps(body).encode() if body is not None else None))
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
            return r.status, json.loads(raw) if r.headers.get("Content-Type") == "application/json" else raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def wait_for(url, run_id, states, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, st = call(url + f"api/state?run={run_id}")
        if st["job"]["state"] in states:
            return st["job"]
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_upload_run_and_download(app):
    url, root = app
    assert call(url + "api/config")[1] == {"uploads": True}
    code, run = call(url + "api/runs", "POST", {"title": "Bridge north"})
    assert code == 201 and run["state"] == "waiting_for_files" and "bridge-north" in run["id"]
    rid = run["id"]

    # cannot start before both files arrive
    code, err = call(url + f"api/runs/{rid}/start", "POST", {})
    assert code == 400 and "missing video and telemetry" in err["error"]

    assert call(url + f"api/runs/{rid}/files/video?name=DJI_0001.MP4", "PUT", data=b"x" * 3_000_000)[0] == 200
    assert call(url + f"api/runs/{rid}/files/telemetry?name=DJI_0001.SRT", "PUT", data=b"1\n")[0] == 200
    assert (root / rid / "input" / "video.mp4").stat().st_size == 3_000_000

    code, st = call(url + f"api/runs/{rid}/start", "POST", {})
    assert code == 200 and st["state"] in ("queued", "running")
    job = wait_for(url, rid, ("done", "failed"))
    assert job["state"] == "done" and job["stages_done"] == ["Ingestion", "Synchronisation"]

    code, body = call(url + f"runs/{rid}/download/model.glb")
    assert code == 200 and body == b"glb!"
    _, runs = call(url + "api/runs")
    assert runs[0]["id"] == rid and runs[0]["video_name"] == "DJI_0001.MP4"


def test_bad_uploads_are_refused(app):
    url, _ = app
    _, run = call(url + "api/runs", "POST", {"title": "x"})
    rid = run["id"]
    code, err = call(url + f"api/runs/{rid}/files/video?name=evil.exe", "PUT", data=b"MZ")
    assert code == 415 and ".mp4" in err["error"]
    assert call(url + f"api/runs/{rid}/files/secrets?name=a.mp4", "PUT", data=b"x")[0] == 404
    assert call(url + "api/runs/..%2F..%2Fetc/files/video?name=a.mp4", "PUT", data=b"x")[0] == 404
    assert call(url + f"runs/{rid}/download/..%2Finput%2Fvideo.mp4")[0] == 404
    assert call(url + f"runs/{rid}/download/status.json")[0] == 404
    assert call(url + "runs/nope/report")[0] == 404


def test_failures_reach_the_page(app):
    url, _ = app
    _, run = call(url + "api/runs", "POST", {"title": "should fail"})
    rid = run["id"]
    call(url + f"api/runs/{rid}/files/video?name=a.mov", "PUT", data=b"x")
    call(url + f"api/runs/{rid}/files/telemetry?name=a.csv", "PUT", data=b"x")
    call(url + f"api/runs/{rid}/start", "POST", {})
    job = wait_for(url, rid, ("done", "failed"))
    assert job["state"] == "failed" and job["message"] == "telemetry has no GPS"
    # once started, a run no longer accepts new files
    code, err = call(url + f"api/runs/{rid}/files/video?name=b.mp4", "PUT", data=b"x")
    assert code == 409 and "already started" in err["error"]


def test_interrupted_runs_are_marked_on_restart(tmp_path):
    jobs = JobQueue(tmp_path / "runs", runner=fake_pipeline)
    job = jobs.create("left running")
    job.write(state="running")
    again = JobQueue(tmp_path / "runs", runner=fake_pipeline)
    st = again.get(job.id).read()
    assert st["state"] == "failed" and "Interrupted" in st["message"]
