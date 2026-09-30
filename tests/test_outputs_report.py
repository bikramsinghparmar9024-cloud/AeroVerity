import copy
import json

import pytest

from aerometric.outputs.report import ReportBuilder, generate_key, load_key, verify_report, write_report


def clock():
    t = iter(f"2026-09-29T10:00:{s:02d}+00:00" for s in range(60))
    return lambda: next(t)


def build(tmp_path, key=None):
    (tmp_path / "out").mkdir(exist_ok=True)
    video = tmp_path / "in.mp4"
    video.write_bytes(b"video bytes")
    glb = tmp_path / "out" / "model.glb"
    glb.write_bytes(b"glb bytes")
    dsm = tmp_path / "out" / "dsm.tif"
    dsm.write_bytes(b"tif bytes")
    rb = ReportBuilder("Demo flight", "run-001", {"seed": 0}, clock=clock())
    rb.add("Ingestion", "Read 1 video and telemetry.", {"frames": 300}, inputs=[video])
    rb.add("Georeferencing", "Placed with GPS and gravity.", {"gps_rms_m": 0.8, "roll_std_deg": float("nan")},
           warnings=["tilt across the flight line is uncertain"])
    rb.add("Outputs", "Wrote mesh and DSM.", outputs=[glb, dsm], root=tmp_path / "out")
    key = key or generate_key(tmp_path / "keys" / "signing.pem")
    doc = rb.finalize(key, {"trust": {"measured": 0.62, "inferred": 0.3, "unknown": 0.08}, "gsd_cm": 2.4})
    return doc, key


def test_valid_report_verifies_including_files(tmp_path):
    doc, key = build(tmp_path)
    check = verify_report(doc, files_root=tmp_path / "out")
    assert check.ok and check.chain_ok and check.signature_ok and check.files_ok
    assert doc["entries"][1]["metrics"]["roll_std_deg"] is None  # NaN made JSON-safe
    assert doc["entries"][2]["outputs"][0]["path"] == "model.glb"
    js, page = write_report(tmp_path / "report", doc)
    assert verify_report(json.loads(js.read_text(encoding="utf-8"))).ok  # survives a disk round trip


@pytest.mark.parametrize("tamper, expected", [
    (lambda d: d["entries"][1]["metrics"].__setitem__("gps_rms_m", 0.1), "was modified"),
    (lambda d: d["entries"].pop(1), "does not follow"),
    (lambda d: d["entries"].pop(), "chain head"),
    (lambda d: d["overview"].__setitem__("gsd_cm", 1.0), "signature"),
])
def test_tampering_is_detected(tmp_path, tamper, expected):
    doc, _ = build(tmp_path)
    bad = copy.deepcopy(doc)
    tamper(bad)
    check = verify_report(bad)
    assert not check.ok and any(expected in p for p in check.problems)


def test_changed_output_file_and_foreign_key(tmp_path):
    doc, key = build(tmp_path)
    (tmp_path / "out" / "dsm.tif").write_bytes(b"edited")
    check = verify_report(doc, files_root=tmp_path / "out")
    assert check.chain_ok and check.signature_ok and check.files_ok is False
    assert any("dsm.tif has changed" in p for p in check.problems)

    # someone re-signs a doctored report with their own key: signature is valid, but not ours
    ours = doc["signature"]["public_key"]
    forged, _ = build(tmp_path, key=generate_key(tmp_path / "keys" / "other.pem"))
    assert verify_report(forged).signature_ok
    assert not verify_report(forged, trusted_public_key=ours).ok


def test_key_round_trip_and_html(tmp_path):
    key = generate_key(tmp_path / "k.pem")
    assert load_key(tmp_path / "k.pem").public_key().public_bytes_raw() == key.public_key().public_bytes_raw()
    doc, _ = build(tmp_path, key)
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><polyline points="0,0 1,1"/></svg>'
    _, page = write_report(tmp_path / "r", doc, svg)
    text = page.read_text(encoding="utf-8")
    for s in ("Demo flight", "Measured 62%", "Tilt across the flight line is uncertain", "Georeferencing",
              "<polyline", doc["chain_head"]):
        assert s in text
    assert "<script" not in text  # static, self-contained
