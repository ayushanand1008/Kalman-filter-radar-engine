"""Vectorized constant-velocity Kalman filter over a whole track bank.

Instead of one 6x6 filter object per track (the C++/Eigen approach), we keep
the entire bank as stacked arrays and let numpy/BLAS batch the matrix algebra:

    x : (K, 6)      state vectors  [x, y, z, vx, vy, vz]
    P : (K, 6, 6)   covariances

`np.einsum` / batched `@` give us the same SIMD-vectorized, temporary-free
matmuls that Eigen's expression templates + `.noalias()` are designed to
produce -- but across every track at once, which is the bigger win.

State layout: position (0:3), velocity (3:6). Measurements are position only.
"""
from __future__ import annotations

import numpy as np

STATE_DIM = 6
MEAS_DIM = 3

# Measurement matrix H (3x6): observe position, not velocity.
H = np.zeros((MEAS_DIM, STATE_DIM))
H[0, 0] = H[1, 1] = H[2, 2] = 1.0
_HT = H.T
_I6 = np.eye(STATE_DIM)


def transition_matrix(dt: float) -> np.ndarray:
    """Constant-velocity state transition F(dt) (6x6)."""
    f = np.eye(STATE_DIM)
    f[0, 3] = f[1, 4] = f[2, 5] = dt
    return f


def process_noise(dt: float, q: float) -> np.ndarray:
    """Discrete white-noise-acceleration process covariance Q (6x6).

    `q` is the power spectral density of the acceleration noise (m^2/s^3).
    Per axis the continuous WNA model discretizes to
        [[dt^3/3, dt^2/2], [dt^2/2, dt]] * q
    coupling that axis's position and velocity.
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    q_pos = dt3 / 3.0
    q_pv = dt2 / 2.0
    q_vel = dt
    qm = np.zeros((STATE_DIM, STATE_DIM))
    for i in range(3):
        p, v = i, i + 3
        qm[p, p] = q_pos
        qm[p, v] = qm[v, p] = q_pv
        qm[v, v] = q_vel
    return qm * q


def predict(x: np.ndarray, P: np.ndarray, dt: float, q: float):
    """Batched predict step.

    x: (K, 6), P: (K, 6, 6) -> propagated (x, P).
    """
    F = transition_matrix(dt)
    Q = process_noise(dt, q)
    x_pred = x @ F.T                      # (K,6) @ (6,6)
    # P' = F P F^T + Q, batched over K
    FP = np.einsum('ij,kjl->kil', F, P)          # (K,6,6)
    P_pred = np.einsum('kil,jl->kij', FP, F) + Q  # (K,6,6) + (6,6) broadcasts
    return x_pred, P_pred


def innovation_covariance(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """S = H P H^T + R for a bank of covariances. Returns (K, 3, 3).

    Because H selects position, H P H^T is simply the top-left 3x3 block of P.
    """
    S = P[:, :MEAS_DIM, :MEAS_DIM] + R
    return S


def update(x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray):
    """Batched Kalman update using the Joseph-form covariance for stability.

    x: (K, 6), P: (K, 6, 6), z: (K, 3) measurements, R: (3, 3) noise.
    Every track in the bank must have an assigned measurement `z[k]`.
    Returns updated (x, P).
    """
    K = x.shape[0]
    if K == 0:
        return x, P

    S = innovation_covariance(P, R)                       # (K,3,3)
    PHt = np.einsum('kij,lj->kil', P, H)                  # (K,6,3) == P @ H^T
    Sinv = np.linalg.inv(S)                               # (K,3,3)
    gain = np.einsum('kil,klm->kim', PHt, Sinv)          # (K,6,3) Kalman gain

    y = z - x[:, :MEAS_DIM]                               # innovation (K,3)
    x_upd = x + np.einsum('kim,km->ki', gain, y)         # (K,6)

    # Joseph form: P = (I-KH) P (I-KH)^T + K R K^T  -- numerically PSD-safe.
    KH = np.einsum('kim,mj->kij', gain, H)               # (K,6,6)
    ImKH = _I6 - KH
    P_upd = (np.einsum('kij,kjl->kil',
                       np.einsum('kij,kjl->kil', ImKH, P),
                       np.transpose(ImKH, (0, 2, 1)))
             + np.einsum('kim,mn,kjn->kij', gain, R, gain))
    return x_upd, P_upd


def mahalanobis_sq(x_pred: np.ndarray, P_pred: np.ndarray,
                   meas: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distances between predicted tracks and measurements.

    x_pred: (K, 6), P_pred: (K, 6, 6), meas: (M, 3).
    Returns (K, M) distances d^2 = y^T S^-1 y, honouring each track's own
    innovation covariance (the "covariance ellipse" / directional momentum).
    """
    S = innovation_covariance(P_pred, R)                 # (K,3,3)
    Sinv = np.linalg.inv(S)                              # (K,3,3)
    # y[k,m] = meas[m] - pos[k]
    pos = x_pred[:, :MEAS_DIM]                           # (K,3)
    y = meas[None, :, :] - pos[:, None, :]              # (K,M,3)
    # d2 = einsum over the quadratic form with each track's Sinv
    tmp = np.einsum('kmi,kij->kmj', y, Sinv)            # (K,M,3)
    d2 = np.einsum('kmj,kmj->km', tmp, y)              # (K,M)
    return d2
