"""Register flight B onto flight A before comparing them.

Each flight is a complete, separate pipeline run with its own local frame.
  1. GPS prior: B's points are carried into A's frame through latitude/longitude.
  2. Feature alignment (optional): the two orthophotos are feature-matched for a
     coarse horizontal shift and heading correction, useful when the GPS of one
     flight is off by more than the ICP can pull in.
  3. ICP: trimmed point-to-plane ICP refines a rigid transform. Only the
     best-fitting share of correspondences is used each iteration, so areas that
     really changed between flights do not drag the alignment.
The result maps B (in A's frame) onto A: X_A = R X_B + t.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from dronerecon.config import ChangeConfig, MatchingConfig
from dronerecon.geo import LocalFrame
from dronerecon.matching.features import make_matching
from dronerecon.pose.types import Similarity
from dronerecon.types import DataError

IDENTITY = Similarity(1.0, np.eye(3), np.zeros(3))


@dataclass(frozen=True)
class FlightRegistration:
    transform: Similarity  # B (already in A's frame via GPS) -> A
    rms_m: float  # trimmed residual after ICP
    overlap: float  # share of B points with a partner in A
    iterations: int
    converged: bool
    prior: str  # "gps" or "gps+ortho"
    warnings: tuple[str, ...] = ()


def to_frame(points: np.ndarray, src: LocalFrame, dst: LocalFrame) -> np.ndarray:
    """Re-express local ENU points of one flight in another flight's local frame."""
    lat, lon, h = src.to_lla(points)
    return dst.to_enu(lat, lon, h)


# --------------------------------------------------------------------------- orthophoto prior

def ortho_prior(ortho_a: np.ndarray, ortho_b: np.ndarray, cell_m: float, origin_a: tuple[float, float],
                offset_b_px: tuple[float, float] = (0, 0), cfg: ChangeConfig | None = None) -> Similarity | None:
    """Horizontal shift + heading between two RGBA orthophotos on grids of the same cell size.

    origin_a: (east, north) of A's pixel (0, 0) centre, in A's frame.
    offset_b_px: where B's pixel (0, 0) sits in A's pixel grid (columns, rows), from the GPS prior.
    Returns the correction for B's points in A's frame, or None if matching was not convincing.
    """
    cfg = cfg or ChangeConfig()
    ext, mat = make_matching(MatchingConfig(backend="opencv", max_dim=max(ortho_a.shape[:2] + ortho_b.shape[:2])))

    def feats(img):
        bgr = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
        empty = img[..., 3] == 0 if img.shape[2] == 4 else np.zeros(img.shape[:2], bool)
        return ext.extract(bgr, cv2.dilate(empty.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool))

    fa, fb = feats(ortho_a), feats(ortho_b)
    m = mat.match(fb, fa)
    if len(m) < cfg.ortho_min_inliers:
        return None
    src = fb.keypoints[m.idx0] + np.asarray(offset_b_px)
    dst = fa.keypoints[m.idx1]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or int(inl.sum()) < cfg.ortho_min_inliers:
        return None
    # pixels -> metres: EN = o + cell * F px with F = diag(1, -1) (rows go south), so
    # EN' = F A F (EN - o) + o + cell * F b
    A = M[:, :2] / np.sqrt(abs(np.linalg.det(M[:, :2])))  # drop any small scale: flights share metric scale
    F = np.diag([1.0, -1.0])
    R2 = F @ A @ F
    o = np.asarray(origin_a, float)
    t2 = o - R2 @ o + cell_m * F @ M[:, 2]
    R = np.eye(3)
    R[:2, :2] = R2
    return Similarity(1.0, R, np.array([t2[0], t2[1], 0.0]))


# --------------------------------------------------------------------------- ICP

def _normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    _, idx = cKDTree(points).query(points, k=min(k, len(points)))
    nb = points[idx] - points[idx].mean(1, keepdims=True)
    _, _, vt = np.linalg.svd(nb, full_matrices=False)
    n = vt[:, -1]
    return n * np.where(n[:, 2:3] < 0, -1, 1)


def icp(src: np.ndarray, dst: np.ndarray, dst_normals: np.ndarray | None = None, init: Similarity = IDENTITY,
        cfg: ChangeConfig | None = None) -> tuple[Similarity, float, float, int, bool]:
    """Trimmed point-to-plane ICP. Returns (transform, rms, overlap, iterations, converged)."""
    cfg = cfg or ChangeConfig()
    if len(src) < 10 or len(dst) < 10:
        raise DataError("ICP: need at least 10 points in each cloud")
    normals = dst_normals if dst_normals is not None else _normals(dst)
    tree = cKDTree(dst)
    R, t = init.R.copy(), init.t.copy()
    rms, overlap, converged = float("inf"), 0.0, False
    it = 0
    for it in range(1, cfg.icp_max_iter + 1):
        P = src @ R.T + t
        d, j = tree.query(P, distance_upper_bound=cfg.icp_max_dist_m)
        ok = np.isfinite(d)
        overlap = float(ok.mean())
        if ok.sum() < 10:
            break
        P, Q, N = P[ok], dst[j[ok]], normals[j[ok]]
        r = np.einsum("ij,ij->i", P - Q, N)
        keep = np.abs(r) <= np.quantile(np.abs(r), cfg.icp_trim)
        P, Q, N, r = P[keep], Q[keep], N[keep], r[keep]
        rms = float(np.sqrt(np.mean(r ** 2)))
        # linearised point-to-plane: [p x n, n] [w; dt] = -r
        A = np.column_stack([np.cross(P, N), N])
        x, *_ = np.linalg.lstsq(A, -r, rcond=None)
        dR = Rotation.from_rotvec(x[:3]).as_matrix()
        R, t = dR @ R, dR @ t + x[3:]
        if np.linalg.norm(x[3:]) < cfg.icp_tol_m and np.linalg.norm(x[:3]) < cfg.icp_tol_m / 100:
            converged = True
            break
    return Similarity(1.0, R, t), rms, overlap, it, converged


def register_flights(points_a: np.ndarray, frame_a: LocalFrame, points_b: np.ndarray, frame_b: LocalFrame,
                     normals_a: np.ndarray | None = None, prior: Similarity | None = None,
                     cfg: ChangeConfig | None = None) -> FlightRegistration:
    """points_*: each flight's surface points in its own local frame. prior: optional ortho correction."""
    cfg = cfg or ChangeConfig()
    b_in_a = to_frame(points_b, frame_b, frame_a)
    init = prior or IDENTITY
    T, rms, overlap, iters, conv = icp(b_in_a, points_a, normals_a, init, cfg)
    warnings: list[str] = []
    if overlap < cfg.min_overlap:
        warnings.append(f"only {overlap:.0%} of flight B overlaps flight A; the comparison covers a small area "
                        "and the alignment may be weak")
    if not conv:
        warnings.append("the alignment did not fully settle; treat small changes with caution")
    shift = float(np.linalg.norm(T.t[:2]))
    heading = float(np.degrees(abs(Rotation.from_matrix(T.R).as_euler("zyx")[0])))
    if shift > 5 or heading > 2:
        warnings.append(f"flight B needed a {shift:.1f} m / {heading:.1f}° correction to line up with flight A; "
                        "the GPS of one flight was noticeably off")
    return FlightRegistration(T, rms, overlap, iters, conv, "gps+ortho" if prior is not None else "gps",
                              tuple(warnings))


def apply(reg: FlightRegistration, points_b: np.ndarray, frame_b: LocalFrame, frame_a: LocalFrame) -> np.ndarray:
    """Flight B points expressed in A's frame, aligned to A."""
    return reg.transform.apply_points(to_frame(points_b, frame_b, frame_a))
