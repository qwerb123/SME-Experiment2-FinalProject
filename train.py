import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares, minimize


def your_algorithm(d_hat_u, p_bs):
    bs = np.asarray(p_bs, dtype=float).T
    r = np.asarray(d_hat_u, dtype=float).ravel()
    tau, kappa = 2.0, 5.0

    valid = np.isfinite(r) & (r > 0)
    if valid.sum() < 3:
        return bs.mean(axis=0)
    use = np.where(valid)[0]
    B, rr = bs[use], r[use]

    step = 2.0
    xlo, xhi = bs[:, 0].min() - 15.0, bs[:, 0].max() + 15.0
    ylo, yhi = bs[:, 1].min() - 15.0, bs[:, 1].max() + 15.0
    gx = np.arange(xlo, xhi + step, step)
    gy = np.arange(ylo, yhi + step, step)
    cells = np.column_stack([m.ravel() for m in np.meshgrid(gx, gy)])

    cdist = np.linalg.norm(cells[:, None, :] - B[None, :, :], axis=2)
    e = rr[None, :] - cdist
    los_match = (np.abs(e) <= tau)
    outside = (e < -tau)
    score = los_match.sum(axis=1) - 3.0 * outside.sum(axis=1)
    x0 = cells[int(np.argmax(score))]

    e0 = rr - np.linalg.norm(B - x0, axis=1)
    sigma = float(np.clip(2.0 * 1.4826 * np.median(np.abs(e0 - np.median(e0))), 3.0, 15.0))

    def nll(x):
        d = rr - np.linalg.norm(B - x, axis=1)
        return np.where(d >= 0, d / sigma, kappa * (-d)).sum()

    x = minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 500, 'maxfev': 500}).x

    e_all = r - np.linalg.norm(bs - x, axis=1)
    inl = np.where((e_all >= -tau) & (e_all <= tau))[0]
    if len(inl) < 3:
        inl = np.where((e_all >= -5.0) & (e_all <= 5.0))[0]
    if len(inl) >= 3:
        x = least_squares(lambda y: np.linalg.norm(bs[inl] - y, axis=1) - r[inl], x).x
    return x


def main():
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    if 'p_bs' in data:
        p_bs = np.asarray(data['p_bs'], dtype=float)
    else:
        p_bs = np.asarray(data['BS_positions'], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)
    return p_hat


if __name__ == "__main__":
    main()
