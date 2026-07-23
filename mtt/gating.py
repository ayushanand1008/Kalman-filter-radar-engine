"""Spatial indexing + Mahalanobis gating.

Comparing every track to every ping is O(K*M). We avoid it in two stages:

1. Coarse spatial query -- a cKDTree (compiled C) over the ping cloud. For each
   track we query a ball whose radius is derived from its innovation
   covariance, instantly discarding physically impossible pairs. In dense,
   bounded airspace a uniform hash grid can beat a K-D tree; a drop-in
   `HashGrid` is provided and both expose `query_radius`.

2. Fine Mahalanobis test -- surviving candidate pairs are scored with the exact
   squared Mahalanobis distance y^T S^-1 y and kept only if below the
   chi-square gate threshold (3 dof).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import kalman

# Chi-square thresholds for 3 degrees of freedom.
CHI2_GATE_95 = 7.815
CHI2_GATE_99 = 11.345


class KDTreeIndex:
    """cKDTree wrapper exposing a per-track variable-radius ball query."""

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        self._tree = cKDTree(self.points) if len(self.points) else None

    def query_radius(self, centers: np.ndarray, radii: np.ndarray):
        """Return a list (len K) of ping-index arrays within each track radius."""
        if self._tree is None or len(centers) == 0:
            return [np.empty(0, dtype=int) for _ in range(len(centers))]
        # query_ball_point supports a per-point radius via an array argument.
        raw = self._tree.query_ball_point(centers, radii)
        return [np.asarray(r, dtype=int) for r in raw]


class HashGrid:
    """Uniform spatial hash grid: O(1) average bucket lookups.

    Provided as a drop-in alternative to `KDTreeIndex` for dense scenes.
    """

    def __init__(self, points: np.ndarray, cell: float = 2000.0):
        self.points = np.asarray(points, dtype=float)
        self.cell = float(cell)
        self._buckets: dict[tuple, list] = {}
        if len(self.points):
            keys = np.floor(self.points / self.cell).astype(np.int64)
            for i, k in enumerate(map(tuple, keys)):
                self._buckets.setdefault(k, []).append(i)

    def query_radius(self, centers: np.ndarray, radii: np.ndarray):
        out = []
        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float)
        for c, r in zip(centers, radii):
            span = int(np.ceil(r / self.cell))
            base = np.floor(c / self.cell).astype(np.int64)
            cand = []
            for dx in range(-span, span + 1):
                for dy in range(-span, span + 1):
                    for dz in range(-span, span + 1):
                        cand.extend(self._buckets.get(
                            (base[0] + dx, base[1] + dy, base[2] + dz), []))
            if cand:
                cand = np.asarray(cand, dtype=int)
                d = np.linalg.norm(self.points[cand] - c, axis=1)
                out.append(cand[d <= r])
            else:
                out.append(np.empty(0, dtype=int))
        return out


def gate(x_pred, P_pred, meas, R, gate_threshold=CHI2_GATE_99, index_cls=KDTreeIndex):
    """Gate tracks against measurements.

    Returns a list (len K) of (ping_indices, d2) tuples: for each track, the
    measurement indices that pass the Mahalanobis gate and their squared
    distances. Uses a covariance-scaled coarse radius, then the exact test.
    """
    K = x_pred.shape[0]
    M = meas.shape[0]
    if K == 0 or M == 0:
        return [(np.empty(0, dtype=int), np.empty(0)) for _ in range(K)]

    S = kalman.innovation_covariance(P_pred, R)          # (K,3,3)
    # Coarse ball radius: sqrt(gate * largest innovation-variance eigenvalue).
    # Using the max eigenvalue guarantees the ball encloses the true gate
    # ellipsoid, so the coarse pass never drops a valid candidate.
    max_eig = np.linalg.eigvalsh(S)[:, -1]               # (K,)
    radii = np.sqrt(gate_threshold * max_eig)

    index = index_cls(meas)
    pos = x_pred[:, :3]
    candidates = index.query_radius(pos, radii)

    Sinv = np.linalg.inv(S)
    results = []
    for k in range(K):
        cand = candidates[k]
        if cand.size == 0:
            results.append((cand, np.empty(0)))
            continue
        y = meas[cand] - pos[k]                           # (c,3)
        d2 = np.einsum('ci,ij,cj->c', y, Sinv[k], y)     # exact Mahalanobis^2
        keep = d2 <= gate_threshold
        results.append((cand[keep], d2[keep]))
    return results
