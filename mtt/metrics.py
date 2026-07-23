"""Validation metrics: score tracker output against synthetic ground truth.

We compute a MOT-style scorecard by matching confirmed track estimates to true
targets frame-by-frame (optimal assignment, gated by distance):

    MOTP  -- mean position error over matched pairs (m)
    MOTA  -- 1 - (misses + false_positives + id_switches) / total_gt
    ID switches -- a truth's matched track id changing between frames
    false tracks -- confirmed tracks matched to no truth
    mostly-tracked fraction -- truths matched in >=80% of their live frames

These require ground truth, which is exactly why the scenarios are synthetic.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(scenario, frame_reports, match_gate=800.0):
    """Score a list of per-frame tracker reports against a scenario's truth.

    frame_reports[i] corresponds to scenario.times[i] and contains a "tracks"
    list of {"id", "status", "pos"} dicts. Only confirmed tracks are scored.
    """
    truth = scenario.truth
    times = scenario.times
    truth_ids = sorted(truth.keys())

    total_gt = 0
    misses = 0
    false_pos = 0
    id_switches = 0
    err_sum = 0.0
    err_cnt = 0

    last_track_for_truth: dict[int, int] = {}
    truth_seen = {tid: 0 for tid in truth_ids}
    truth_matched = {tid: 0 for tid in truth_ids}

    for fi, report in enumerate(frame_reports):
        # present truths at this frame
        present = []
        present_pos = []
        for tid in truth_ids:
            p = truth[tid][fi]
            if not np.any(np.isnan(p)):
                present.append(tid)
                present_pos.append(p)
                truth_seen[tid] += 1
        total_gt += len(present)
        present_pos = np.asarray(present_pos).reshape(-1, 3)

        conf = [t for t in report["tracks"] if t["status"] == "confirmed"]
        est_ids = [t["id"] for t in conf]
        est_pos = np.asarray([t["pos"] for t in conf]).reshape(-1, 3)

        if len(present) and len(conf):
            d = np.linalg.norm(present_pos[:, None, :] - est_pos[None, :, :], axis=2)
            BIG = 1e9
            cost = np.where(d <= match_gate, d, BIG)
            rows, cols = linear_sum_assignment(cost)
            matched_truth = set()
            matched_est = set()
            for r, c in zip(rows, cols):
                if cost[r, c] >= BIG:
                    continue
                tid = present[r]
                eid = est_ids[c]
                matched_truth.add(r)
                matched_est.add(c)
                err_sum += d[r, c]
                err_cnt += 1
                truth_matched[tid] += 1
                if tid in last_track_for_truth and last_track_for_truth[tid] != eid:
                    id_switches += 1
                last_track_for_truth[tid] = eid
            misses += len(present) - len(matched_truth)
            false_pos += len(conf) - len(matched_est)
        else:
            misses += len(present)
            false_pos += len(conf)

    motp = (err_sum / err_cnt) if err_cnt else float("nan")
    mota = 1.0 - (misses + false_pos + id_switches) / max(total_gt, 1)
    mostly_tracked = sum(
        1 for tid in truth_ids
        if truth_seen[tid] and truth_matched[tid] / truth_seen[tid] >= 0.8)

    return {
        "n_targets": len(truth_ids),
        "total_gt_points": total_gt,
        "motp_m": motp,
        "mota": mota,
        "misses": misses,
        "false_tracks": false_pos,
        "id_switches": id_switches,
        "mostly_tracked": mostly_tracked,
        "mostly_tracked_frac": mostly_tracked / max(len(truth_ids), 1),
    }
