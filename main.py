import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares, minimize


def your_algorithm(d_hat_u, p_bs):
    bs = np.asarray(p_bs, dtype=float).T          # (18, 2)
    r = np.asarray(d_hat_u, dtype=float).ravel()  # (18,)
    tau, kappa = 2.0, 5.0

    valid = np.isfinite(r) & (r > 0)
    if valid.sum() < 3:
        return bs.mean(axis=0)
    use = np.where(valid)[0]
    B, rr = bs[use], r[use]

    # --- 1단계: 전역 원판 투표 ---
    # 격자 범위를 기지국 좌표 범위 + 15m 여유로 자동 설정(격자 밖 사용자 포함), 간격 2m
    step = 2.0
    xlo, xhi = bs[:, 0].min() - 15.0, bs[:, 0].max() + 15.0
    ylo, yhi = bs[:, 1].min() - 15.0, bs[:, 1].max() + 15.0
    gx = np.arange(xlo, xhi + step, step)
    gy = np.arange(ylo, yhi + step, step)
    cells = np.column_stack([m.ravel() for m in np.meshgrid(gx, gy)])  # (M, 2)

    cdist = np.linalg.norm(cells[:, None, :] - B[None, :, :], axis=2)  # (M, nuse)
    e = rr[None, :] - cdist          # >0: 칸이 측정 원판 안(단방향에서 정상), 0근처: LOS 경계 일치
    los_match = (np.abs(e) <= tau)   # LOS 경계와 일치하는 측정 수
    outside = (e < -tau)             # 칸이 측정 원판 밖(단방향 모델에서 모순) → 벌점
    score = los_match.sum(axis=1) - 3.0 * outside.sum(axis=1)
    x0 = cells[int(np.argmax(score))]

    # --- σ 자동 결정 (사용자별 잔차의 robust 산포) ---
    e0 = rr - np.linalg.norm(B - x0, axis=1)
    sigma = float(np.clip(2.0 * 1.4826 * np.median(np.abs(e0 - np.median(e0))), 3.0, 15.0))

    # --- 2단계: 지수분포 단방향 최대우도 정밀화 ---
    def nll(x):
        d = rr - np.linalg.norm(B - x, axis=1)   # >0: NLOS 편향(예상), <0: 모델상 불가능
        return np.where(d >= 0, d / sigma, kappa * (-d)).sum()

    x = minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 500, 'maxfev': 500}).x

    # --- 3단계: LOS 인라이어만으로 재추정 (인라이어 부족 시 기준 완화) ---
    e_all = r - np.linalg.norm(bs - x, axis=1)
    inl = np.where((e_all >= -tau) & (e_all <= tau))[0]
    if len(inl) < 3:                       # 순수 LOS 링크가 부족한 사용자 방어
        inl = np.where((e_all >= -5.0) & (e_all <= 5.0))[0]
    if len(inl) >= 3:
        x = least_squares(lambda y: np.linalg.norm(bs[inl] - y, axis=1) - r[inl], x).x
    return x


def main():
    # 채점기가 같은 폴더에 둔 .mat 로드 (키 이름 방어: p_bs / BS_positions 모두 시도)
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    if 'p_bs' in data:
        p_bs = np.asarray(data['p_bs'], dtype=float)
    else:
        p_bs = np.asarray(data['BS_positions'], dtype=float)   # (2, 18)
    d_hat = np.asarray(data['d_hat'], dtype=float)             # (18, num_user)

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)
    return p_hat


if __name__ == "__main__":
    main()
