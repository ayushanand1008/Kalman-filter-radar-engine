"""Data association: cost matrix, sub-graph fracturing, optimal assignment.

Given the gated (track -> candidate pings) lists, we assign at most one ping to
each track so total cost is minimized. The global assignment matrix is
sparse and block-structured: most tracks only contend for pings near them.

Two optimizations from the spec:

* Innovation-covariance cost -- cost = Mahalanobis^2 + log|S| (the negative
  log-likelihood of the association under a Gaussian innovation), so the
  directional covariance ellipse, not raw Euclidean distance, drives matching.

* Sub-graph fracturing -- the bipartite gating graph is split into connected
  components via union-find. Each isolated cluster is solved independently with
  scipy's Jonker-Volgenant / Hungarian solver (`linear_sum_assignment`),
  turning one O((K+M)^3) solve into many tiny ones.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _components(n_tracks, n_meas, gated):
    """Union-find over the bipartite graph; yield (track_ids, meas_ids) clusters.

    Nodes 0..n_tracks-1 are tracks, n_tracks..n_tracks+n_meas-1 are pings.
    Tracks with no gated pings are skipped (they will simply coast).
    """
    uf = _UnionFind(n_tracks + n_meas)
    for k, (cand, _d2) in enumerate(gated):
        for m in cand:
            uf.union(k, n_tracks + int(m))

    clusters: dict[int, tuple[list, list]] = {}
    for k, (cand, _d2) in enumerate(gated):
        if cand.size == 0:
            continue
        root = uf.find(k)
        t_list, m_list = clusters.setdefault(root, ([], []))
        t_list.append(k)
    for m in range(n_meas):
        root = uf.find(n_tracks + m)
        if root in clusters:
            clusters[root][1].append(m)
    return list(clusters.values())


def associate(gated, gate_threshold, use_loglik=True, logdetS=None):
    """Compute optimal track<->ping assignments.

    Parameters
    ----------
    gated : list of (cand_ping_idx, d2) per track (output of gating.gate)
    gate_threshold : cap used as the non-assignment cost
    use_loglik : add 0.5*log|S| so the cost is the Gaussian NLL, not just d2
    logdetS : (K,) precomputed log|S| per track (required if use_loglik)

    Returns
    -------
    matches : (P, 2) int array of [track_idx, ping_idx]
    unmatched_tracks : (U,) int array of track indices with no assignment
    unmatched_pings  : (V,) int array of ping indices assigned to no track
    """
    n_tracks = len(gated)
    all_pings = set()
    for cand, _ in gated:
        all_pings.update(int(m) for m in cand)
    n_meas = (max(all_pings) + 1) if all_pings else 0

    matched_tracks = set()
    matched_pings = set()
    matches = []

    # Big-M for forbidden (un-gated) pairs inside a cluster's dense matrix.
    BIG = gate_threshold * 1e6 + 1.0

    for t_list, m_list in _components(n_tracks, n_meas, gated):
        if not t_list or not m_list:
            continue
        t_arr = np.asarray(sorted(t_list))
        m_arr = np.asarray(sorted(m_list))
        m_pos = {int(m): j for j, m in enumerate(m_arr)}

        cost = np.full((len(t_arr), len(m_arr)), BIG)
        for i, k in enumerate(t_arr):
            cand, d2 = gated[k]
            for c, dd in zip(cand, d2):
                j = m_pos[int(c)]
                base = float(dd)
                if use_loglik and logdetS is not None:
                    base += logdetS[k]
                cost[i, j] = base

        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] >= BIG:      # forbidden pair chosen only as filler
                continue
            k = int(t_arr[r])
            m = int(m_arr[c])
            matches.append((k, m))
            matched_tracks.add(k)
            matched_pings.add(m)

    matches = (np.asarray(matches, dtype=int).reshape(-1, 2)
               if matches else np.empty((0, 2), dtype=int))
    unmatched_tracks = np.asarray(
        [k for k in range(n_tracks) if k not in matched_tracks], dtype=int)
    unmatched_pings = np.asarray(
        [m for m in range(n_meas) if m not in matched_pings], dtype=int)
    return matches, unmatched_tracks, unmatched_pings
