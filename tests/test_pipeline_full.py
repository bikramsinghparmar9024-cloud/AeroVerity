import json

import numpy as np
import pytest
import trimesh

from dronerecon.config import FusionConfig, PipelineConfig, UncertaintyConfig
from dronerecon.ingest.flight import FlightInputs
from dronerecon.outputs.report import verify_report
from dronerecon.pipeline.back import run_back
from dronerecon.pipeline.front import run_front
from synthetic_flight import FRAME, make_flight


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    folder = tmp_path_factory.mktemp("full")
    fl = make_flight(folder)
    cfg = PipelineConfig(fusion=FusionConfig(voxel_m=0.5), uncertainty=UncertaintyConfig(n_runs=3, min_valid_runs=2))
    run = folder / "run"
    front = run_front(FlightInputs(fl.video, fl.srt, intrinsics=fl.intrinsics), run, cfg, title="Synthetic flight")
    back = run_back(front, run, cfg, title="Synthetic flight")
    return fl, front, back, run


def test_every_product_is_written(full):
    _, _, back, run = full
    for name in ("glb", "mesh_ply", "sparse_ply", "las", "splat", "tileset", "dsm", "trust", "class", "sigma",
                 "ortho", "classes_geojson", "trust_geojson", "refly", "report_json", "report_html"):
        assert back.files[name].is_file() and back.files[name].stat().st_size > 0, name
    ts = json.loads(back.files["tileset"].read_text())
    assert ts["root"]["children"]


def test_surface_sits_on_the_ground(full):
    _, front, back, _ = full
    lat, lon, h = front.track.frame.to_lla(back.mesh.vertices)
    ground = FRAME.to_enu(lat, lon, h)[:, 2]  # truth ground is z = 0 in the generator's frame
    assert abs(np.median(ground)) < 0.5
    g = trimesh.load(back.files["glb"])
    mesh = list(g.geometry.values())[0] if hasattr(g, "geometry") else g
    assert len(mesh.faces) == len(back.mesh.faces)


def test_trust_uncertainty_and_signed_report(full):
    _, _, back, run = full
    fr = back.trust.fractions(back.mesh.vertices, back.mesh.faces)
    assert sum(fr.values()) == pytest.approx(1.0) and fr["measured"] > 0.2
    assert back.trust.evidence_is_mock  # stand-in depth/segmentation must be declared
    doc = json.loads(back.files["report_json"].read_text(encoding="utf-8"))
    stages = [e["stage"] for e in doc["entries"]]
    assert stages == ["Ingestion", "Synchronisation", "Pre-flight check", "Keyframes", "Matching", "Calibration",
                      "Camera poses", "Coverage", "Georeferencing", "Depth", "Surface", "Uncertainty", "Trust",
                      "Re-fly advice", "Outputs"]
    print("\nOVERVIEW", doc["overview"])
    check = verify_report(doc, files_root=run)
    assert check.ok, check.problems
    assert "gravity" in doc["overview"]["anchors"]
