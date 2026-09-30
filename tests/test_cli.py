import json
import shutil

import numpy as np
import pytest

from dronerecon.cli import main
from dronerecon.geo import LocalFrame
from dronerecon.pipeline.back import load_state
from synthetic_flight import make_flight


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cli")
    fl = make_flight(folder, duration_s=8.0)
    cfg = folder / "fast.json"
    cfg.write_text(json.dumps({"fusion": {"voxel_m": 0.5}, "uncertainty": {"n_runs": 2, "min_valid_runs": 2}}))
    out = folder / "runA"
    code = main(["run", "--video", str(fl.video), "--telemetry", str(fl.srt), "--intrinsics", str(fl.intrinsics),
                 "--config", str(cfg), "--out", str(out), "--title", "CLI flight", "--key", str(folder / "key.pem")])
    return code, folder, out, cfg


def test_run_command(cli_run, capsys):
    code, folder, out, _ = cli_run
    assert code == 0
    assert (out / "report" / "quality_report.html").is_file()
    assert (out / "products" / "state.npz").is_file() and (out / "products" / "model.glb").is_file()


def test_verify_command_passes_then_catches_tampering(cli_run, capsys):
    _, folder, out, _ = cli_run
    rpt = out / "report" / "quality_report.json"
    assert main(["verify", str(rpt), "--files", str(out)]) == 0
    assert "every output file matches" in capsys.readouterr().out
    (out / "products" / "dsm.tif").write_bytes(b"tampered")
    assert main(["verify", str(rpt), "--files", str(out)]) == 1
    assert "dsm.tif has changed" in capsys.readouterr().out


def test_compare_command_finds_a_new_building(cli_run, capsys):
    _, folder, out, cfg = cli_run
    a = load_state(out / "products" / "state.npz")
    # Flight B: same scene, its own frame (origin 2.5 m off, as if GPS differed), plus a new 10 m block
    run_b = folder / "runB"
    (run_b / "products").mkdir(parents=True)
    b_frame = LocalFrame(a.frame.lat0 + 1.5e-5, a.frame.lon0 + 1.5e-5, a.frame.alt0)
    lat, lon, h = a.frame.to_lla(a.vertices)
    vb = b_frame.to_enu(lat, lon, h)
    c = np.median(a.vertices[:, :2], axis=0)
    block = (np.abs(a.vertices[:, 0] - c[0]) < 8) & (np.abs(a.vertices[:, 1] - c[1]) < 8)
    vb[block, 2] += 10.0
    np.savez_compressed(run_b / "products" / "state.npz", vertices=vb, faces=a.faces, normals=a.normals,
                        trust=a.attrs.trust, sigma=a.attrs.sigma_m, classes=a.attrs.labels,
                        frame=np.array([b_frame.lat0, b_frame.lon0, b_frame.alt0]), cell_m=np.array(a.cell_m))
    code = main(["compare", str(out), str(run_b), "--out", str(folder / "change"), "--config", str(cfg)])
    printed = capsys.readouterr().out
    assert code == 0 and "Change map" in printed
    summary = json.loads((folder / "change" / "change_summary.json").read_text())
    assert summary["comparable_fraction"] > 0.3
    assert summary["raised_area_m2"] > 50  # the 16 x 16 m block, where both flights are comparable
    assert summary["lowered_area_m2"] < 0.2 * summary["raised_area_m2"]  # the shifted origin is not "change"
    assert main(["verify", str(folder / "change" / "report" / "quality_report.json")]) == 0


def test_console_on_a_legacy_code_page_does_not_crash(monkeypatch):
    import io
    import sys

    from dronerecon.cli import ConsoleReport, _safe_console

    raw = io.BytesIO()
    legacy = io.TextIOWrapper(raw, encoding="cp1252")  # what Windows uses for redirected output
    monkeypatch.setattr(sys, "stdout", legacy)
    _safe_console()
    ConsoleReport("t", "r", {}).add("Camera poses", "positions sit 5.1σ from GPS; 300 m² at 12°")
    legacy.flush()
    assert b"5.1?" in raw.getvalue() and b"12\xb0" in raw.getvalue()  # sigma replaced, degree kept


def test_missing_input_is_a_readable_error(tmp_path, capsys):
    code = main(["run", "--video", str(tmp_path / "none.mp4"), "--telemetry", str(tmp_path / "none.srt"),
                 "--out", str(tmp_path / "x")])
    assert code == 1
    assert "Could not continue" in capsys.readouterr().err
