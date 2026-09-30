import json

import numpy as np
import pytest
import tifffile

from dronerecon.change.changemap import ChangeClass, compute_change, describe, write_change
from dronerecon.geo import LocalFrame
from dronerecon.outputs.mesh_io import SurfaceAttributes
from dronerecon.outputs.rasters import Grid, SurfaceRasters, rasterize_surface
from dronerecon.config import RasterConfig
from dronerecon.trust import TrustLabel

FRAME = LocalFrame(28.6, 77.2, 210.0)
GRID = Grid(e0=-50.0, n0=50.0, cell_m=1.0, width=101, height=101)


def ground():
    E, N = GRID.centres()
    return 2 * np.sin(E / 15) + 0.02 * N


def rasters(dsm, sigma=0.1, trust=TrustLabel.MEASURED):
    s = np.full(dsm.shape, sigma, np.float32) if np.isscalar(sigma) else sigma.astype(np.float32)
    t = np.full(dsm.shape, trust, np.uint8) if np.isscalar(trust) else trust
    return SurfaceRasters(GRID, dsm.astype(np.float32), t, np.full(dsm.shape, 255, np.uint8), s,
                          np.zeros(dsm.shape, bool))


def scenario(seed=0):
    rng = np.random.default_rng(seed)
    E, N = GRID.centres()
    base = ground()
    a = base + rng.normal(0, 0.1, base.shape)
    b = base + rng.normal(0, 0.1, base.shape)
    building = (np.abs(E + 20) < 10) & (np.abs(N - 15) < 10)  # 20 x 20 m, +10 m
    pit = (np.abs(E - 25) < 5) & (np.abs(N + 20) < 5)  # 10 x 10 m, -3 m
    speck = (np.abs(E - 5) < 1) & (np.abs(N - 5) < 1)  # 2 x 2 m blip: below the area limit
    b = b + np.where(building, 10.0, 0) - np.where(pit, 3.0, 0) + np.where(speck, 2.0, 0)
    return a, b, building, pit


def test_real_changes_found_noise_ignored():
    a, b, building, pit = scenario()
    cm = compute_change(rasters(a), rasters(b), registration_rms_m=0.05)
    kinds = sorted(x.kind for x in cm.areas)
    assert kinds == ["lowered", "raised"]
    raised = next(x for x in cm.areas if x.kind == "raised")
    lowered = next(x for x in cm.areas if x.kind == "lowered")
    cell = GRID.cell_m ** 2
    assert raised.volume_m3 == pytest.approx(10 * building.sum() * cell, rel=0.02)  # 19 x 19 cells
    assert lowered.volume_m3 == pytest.approx(-3 * pit.sum() * cell, rel=0.02)  # 9 x 9 cells
    assert raised.area_m2 == pytest.approx(building.sum() * cell)
    assert raised.center_en == pytest.approx((-20.5, 14.5), abs=1.0)
    quiet = ~(building | pit)
    assert (cm.classes[quiet] == ChangeClass.NO_CHANGE).mean() > 0.99
    assert cm.summary["volume_added_m3"] == pytest.approx(raised.volume_m3)
    text = " ".join(describe(cm))
    assert "rose by up to 10." in text and "dropped by up to 3." in text


def test_uncertain_surfaces_do_not_report_change():
    a, b, building, _ = scenario()
    sigma_b = np.where(building, 6.0, 0.1)  # B barely measured the new building
    cm = compute_change(rasters(a), rasters(b, sigma=sigma_b))
    assert not any(x.kind == "raised" for x in cm.areas)  # 10 m < 2 x 6 m: not claimed


def test_unknown_and_missing_are_not_comparable():
    a, b, building, _ = scenario()
    trust_b = np.where(building, TrustLabel.UNKNOWN, TrustLabel.MEASURED).astype(np.uint8)
    a2 = a.copy()
    a2[:, :10] = np.nan  # flight A did not cover the western strip
    cm = compute_change(rasters(a2), rasters(b, trust=trust_b))
    assert (cm.classes[building] == ChangeClass.NOT_COMPARABLE).all()
    assert (cm.classes[:, :10] == ChangeClass.NOT_COMPARABLE).all()
    assert cm.comparable_fraction < 0.9


def test_grid_mismatch_refused():
    a, b, _, _ = scenario()
    other = SurfaceRasters(Grid(0, 0, 2.0, 101, 101), b.astype(np.float32), np.zeros(b.shape, np.uint8),
                           np.zeros(b.shape, np.uint8), np.zeros(b.shape, np.float32), np.zeros(b.shape, bool))
    with pytest.raises(Exception, match="same grid"):
        compute_change(rasters(a), other)


def test_flight_b_mesh_onto_a_grid_and_files(tmp_path):
    # B arrives as a registered mesh; rasterise it on A's grid, then compare and export
    xs = np.arange(-50, 50.5, 0.5)
    X, Y = np.meshgrid(xs, xs)
    Z = 2 * np.sin(X / 15) + 0.02 * Y + np.where((np.abs(X + 20) < 10) & (np.abs(Y - 15) < 10), 10.0, 0)
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    n = len(xs)
    F = np.array([[j * n + i, j * n + i + 1, (j + 1) * n + i] for j in range(n - 1) for i in range(n - 1)])
    rb = rasterize_surface(V, F, SurfaceAttributes(trust=np.zeros(len(V), np.uint8), sigma_m=np.full(len(V), 0.1)),
                           RasterConfig(fill_holes_px=2), grid=GRID)
    cm = compute_change(rasters(ground()), rb)
    assert [x.kind for x in cm.areas] == ["raised"]
    files = write_change(tmp_path, cm, rasters(ground()), FRAME)
    assert tifffile.imread(files["classes"]).dtype == np.uint8
    gj = json.loads(files["areas"].read_text())
    f = gj["features"][0]["properties"]
    # a DSM keeps the highest point per cell, so cells straddling the walls take the roof height:
    # the block measures between 19 x 19 and 21 x 21 one-metre cells
    assert f["change"] == "raised" and 10 * 19 ** 2 <= f["volume_m3"] <= 10 * 21 ** 2
