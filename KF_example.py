# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import control
plt.rcParams['font.sans-serif'] = ['Yu Gothic'] 
# ----- シミュレーション設定 -----
stepend = 500
steps = np.arange(1, stepend + 1)

# 状態変数（真の状態）・推定状態・観測値の初期化
x = np.zeros((stepend, 2))
x[0] = [0, 0.1]  # 初期状態
hatx = np.zeros((stepend, 2))
hatx[0] = [0, 0]  # 初期推定状態
y = np.zeros(stepend)

# 1ステップ分の誤差分散共分散行列（初期値）
P_k = np.zeros((2, 2))

# 各ステップ毎の事後誤差共分散行列 P と事前誤差共分散行列 P_prior
P = np.zeros((stepend, 2, 2))
P[0] = P_k.copy()
P_prior = np.zeros((stepend, 2, 2))
P_prior[0] = P_k.copy()

# プロセスノイズと観測ノイズの設定
sigmav = 0.1
sigmaw = 0.5
Q = sigmav ** 2
R = sigmaw ** 2

Ts = 0.01
A = np.array([[1, Ts],
              [0,  1]])
b = np.array([[0.2],
              [1]])
c = np.array([[1],
              [0]])

np.random.seed(5)

# ----- (2) オフライン計算：MATLABの kalman 関数相当の計算（比較用） -----
# システム： x[k+1] = A*x[k] + b*v[k]  ,  y[k] = c' * x[k] + w[k]
# control.dlqe で定常カルマンゲインと誤差共分散行列を求める
L_kalmf, P_kalmf, eigVals = control.dlqe(A, b, c.T, Q, R)
Z_kalmf = P_kalmf.copy()  # MATLAB の Z_kalmf の代用（比較用）

# ----- 時変カルマンゲインの配列 -----
K = np.zeros((stepend, 2))

# 推定誤差と実測誤差から計算した誤差共分散行列を保存するための配列
e_stored = np.zeros((stepend, 2))
P_act_stored = np.zeros((stepend, 2, 2))
score = 0.0
sigmaP = np.zeros((stepend, 2))

# ----- (1) オフライン計算による時変ゲイン・誤差共分散行列の計算（検証用） -----
P_bef = np.zeros((stepend, 2, 2))
K_bef = np.zeros((stepend, 2))
for k in range(1, stepend):
    P_bef[k] = A @ P_bef[k - 1] @ A.T + (b @ b.T) * Q
    denom = (c.T @ P_bef[k] @ c + R)[0, 0]
    K_bef[k] = (P_bef[k] @ c).flatten() / denom
    I = np.eye(2)
    P_bef[k] = (I - np.outer(K_bef[k], c.flatten())) @ P_bef[k] @ (I - np.outer(K_bef[k], c.flatten())).T + np.outer(K_bef[k], K_bef[k]) * R

# ----- (3) 定常カルマンフィルタ（定常ゲイン）による推定 -----
hatx_fix = np.zeros((stepend, 2))
hatx_fix[0] = [0, 0]
K_fix = K_bef[-1].copy()  # 定常カルマンゲイン（シミュレーション後半の値）

# ----- シミュレーションループ -----
for k in range(1, stepend):
    # 観測対象（真のシステム）からサンプリング
    x[k] = A.dot(x[k - 1]) + b.flatten() * (np.random.randn() * sigmav)
    y[k] = c.flatten().dot(x[k]) + np.random.randn() * sigmaw

    # 定常カルマンフィルタによる更新
    hatx_fix[k] = A.dot(hatx_fix[k - 1]) + K_fix * (y[k] - c.flatten().dot(A.dot(hatx_fix[k - 1])))

    # 予測ステップ
    hatx_pred = A.dot(hatx[k - 1])

    # 事前誤差共分散行列の更新
    P_prior[k] = A.dot(P[k - 1]).dot(A.T) + (b @ b.T) * Q
    P_k_current = P_prior[k].copy()

    # カルマンゲインの算出
    denom = (c.T @ P_k_current @ c + R)[0, 0]
    K[k] = (P_k_current @ c).flatten() / denom
    K_k = K[k].copy()

    # 更新ステップ
    hatx[k] = hatx_pred + K_k * (y[k] - c.flatten().dot(hatx_pred))

    # 事後誤差共分散行列の更新（Joseph の形）
    I = np.eye(2)
    P[k] = (I - np.outer(K_k, c.flatten())) @ P_k_current @ (I - np.outer(K_k, c.flatten())).T + np.outer(K_k, K_k) * R

    # 推定誤差とスコア更新
    e_k = x[k] - hatx[k]
    P_act_stored[k] = np.outer(e_k, e_k)
    score += np.trace(P_act_stored[k])
    sigmaP[k] = np.sqrt(np.diag(P[k]))

# ----- 推定結果のプロット -----
# ① 位置推定
plt.figure()
plt.plot(steps, x[:, 0], '-b', label='真の位置')
plt.plot(steps, y, '.r', label='観測値y')
plt.plot(steps, hatx[:, 0], '-m', label='推定位置')
plt.plot(steps, hatx[:, 0] + 2 * sigmaP[:, 0], '-g', label='推定位置±2σ')
plt.plot(steps, hatx[:, 0] - 2 * sigmaP[:, 0], '-g')
plt.legend()
plt.title('位置推定の様子')
plt.ylim([-2, 2])

# ② 速度推定
plt.figure()
plt.plot(steps, x[:, 1], '-b', label='真の速度')
plt.plot(steps, hatx[:, 1], '-m', label='推定速度')
plt.plot(steps, hatx[:, 1] + 2 * sigmaP[:, 1], '-g', label='推定速度±2σ')
plt.plot(steps, hatx[:, 1] - 2 * sigmaP[:, 1], '-g')
plt.legend()
plt.title('速度推定の様子')
plt.ylim([-2, 2])

# ±2σからの逸脱率
lowerBound = hatx[:, 0] - 2 * sigmaP[:, 0]
upperBound = hatx[:, 0] + 2 * sigmaP[:, 0]
numOutside = np.sum((x[:, 0] < lowerBound) | (x[:, 0] > upperBound))
print('推定位置が±2σから逸脱した割合: {:.2f}'.format(1 - numOutside / stepend))

# ----- 移動平均による実際の誤差共分散行列の算出 -----
window = 10  # 移動平均のウィンドウサイズ
P_act_movavg = np.zeros((stepend, 2, 2))
for k in range(stepend):
    if k < window:
        P_act_movavg[k] = np.mean(P_act_stored[:k + 1], axis=0)
    else:
        P_act_movavg[k] = np.mean(P_act_stored[k - window + 1:k + 1], axis=0)

# ----- 誤差共分散行列の各要素のプロット -----
# プロット対象： P(1,1), P(1,2), P(2,2) （MATLAB では (2,1) は省略）
plotElems = np.array([[1, 1],
                      [1, 2],
                      [2, 2]])
subPositions = [1, 2, 4]  # 2×2 のサブプロットの位置 (1, 2, 4)

plt.figure()
for n in range(plotElems.shape[0]):
    i1 = int(plotElems[n, 0]) - 1  # Python の 0 始まりに合わせる
    i2 = int(plotElems[n, 1]) - 1
    plt.subplot(2, 2, subPositions[n])
    plt.grid(True)
    # カルマンフィルタで予測された P（緑）
    plt.plot(P[:, i1, i2], '-g', linewidth=1, label='Predicted(KF)')
    # 実際の誤差から計算した P（赤）
    plt.plot(P_act_movavg[:, i1, i2], '-r', linewidth=1, label='Actual')
    plt.title('P({},{})'.format(i1 + 1, i2 + 1))
    plt.xlabel('step')
    plt.ylabel('(x-hatx)^2')
    plt.legend(loc='upper right')
    plt.ylim([0, 0.3])
# 空白にするサブプロット（MATLAB では subplot(2,2,3) を空ける）
plt.subplot(2, 2, 3)
plt.axis('off')
plt.tight_layout()

print('Pの対角要素の累積(スコア):', score)

# ----- オフライン計算とシミュレーション内の計算結果の検証 -----
if np.allclose(K_bef, K) and np.allclose(P_bef, P):
    print('KとPの計算結果は、事前計算・シミュレーションループ内での計算どちらも一致。')

# ----- カルマンゲインの時系列の表示 -----
plt.figure()
plt.plot(steps, K[:, 0], label='K(1)')
plt.plot(steps, K[:, 1], label='K(2)')
plt.legend()
plt.title('カルマンゲインの時間変化')

# ----- 推定誤差の平均（位置・速度）を表示 -----
print('位置推定誤差の平均:', np.mean(x[:, 0] - hatx[:, 0]))
print('速度推定誤差の平均:', np.mean(x[:, 1] - hatx[:, 1]))

# ----- 時変カルマンフィルタと定常カルマンフィルタの比較 -----
fig, axs = plt.subplots(2, 1, tight_layout=True)
axs[0].plot(steps, x[:, 0] - hatx[:, 0], '-m', label='位置誤差(時変)')
axs[0].plot(steps, x[:, 0] - hatx_fix[:, 0], '-c', label='位置誤差(定常)')
axs[0].legend()
axs[0].set_title('時変カルマンフィルタと定常カルマンフィルタの位置推定誤差の比較')
axs[1].plot(steps, x[:, 1] - hatx[:, 1], '-m', label='速度誤差(時変)')
axs[1].plot(steps, x[:, 1] - hatx_fix[:, 1], '-c', label='速度誤差(定常)')
axs[1].legend()
axs[1].set_title('時変カルマンフィルタと定常カルマンフィルタの速度推定誤差の比較')

print('kalman関数でのPとシミュレーション結果のP(の定常値)の差:', np.trace(Z_kalmf - P[-1]))

# ----- カルマンフィルタの伝達関数による位置・速度推定 -----
# 定常時のゲインを用いる（シミュレーション最終時刻の値）
k1 = K[-1, 0]
k2 = K[-1, 1]

# 伝達関数 hatp_flt: (k1*(z-1)+k2*Ts)*z / ((z-1)**2 + (k1+k2*Ts)*(z-1) + k2*Ts)
# 伝達関数 hatq_flt: k2*(z-1)*z / ((z-1)**2 + (k1+k2*Ts)*(z-1) + k2*Ts)
# ※ z-1 は多項式 [1, -1] で表現する。  
num_hatp = [k1, -k1 + k2 * Ts, 0]  # z^2, z^1, z^0 の係数（(k1*z - k1 + k2*Ts)*z）
den_hatp = [1 + k1 + k2 * Ts, -2 - k1 - k2 * Ts, 1 + k2 * Ts]  # (z-1)^2 + (k1+k2*Ts)*(z-1) + k2*Ts
hatp_flt = control.TransferFunction(num_hatp, den_hatp, Ts)

num_hatq = [k2, -k2, 0]  # k2*(z-1)*z
den_hatq = den_hatp.copy()
hatq_flt = control.TransferFunction(num_hatq, den_hatq, Ts)

# 測定値 y に対する伝達関数の応答をシミュレーション（lsim 相当）
t = np.arange(0, stepend * Ts, Ts)
t, psim = control.forced_response(hatp_flt, T=t, U=y)
t, qsim = control.forced_response(hatq_flt, T=t, U=y)

# 伝達関数の応答と定常カルマンフィルタの推定結果の比較
fig, axs = plt.subplots(2, 1, tight_layout=True)
axs[0].plot(t, psim, label='位置誤差(時変)')
axs[0].plot(t, hatx_fix[:, 0], label='位置誤差(定常)')
axs[0].legend()
axs[1].plot(t, qsim, label='速度誤差(時変)')
axs[1].plot(t, hatx_fix[:, 1], label='速度誤差(定常)')
axs[1].legend()
print('max(|psim - hatx_fix(位置)|):', np.max(np.abs(psim - hatx_fix[:, 0])))
print('max(|qsim - hatx_fix(速度)|):', np.max(np.abs(qsim - hatx_fix[:, 1])))

# Bode プロット（伝達関数の周波数応答）
plt.figure()
control.bode_plot([hatp_flt, hatq_flt], dB=True)
plt.grid(True)

plt.show()
