# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D プロット用
from scipy.stats import multivariate_normal

plt.rcParams['font.sans-serif'] = ['Yu Gothic'] 
#===========================
#  パラメータ設定   
#===========================
mu_x = 1
mu_y = 1
sigma_x = 1
sigma_y = np.sqrt(2)  # または sigma_y = 0.8 とする場合もあり
rho = 0.8

mu = np.array([mu_x, mu_y])
# 分散共分散行列の作成
Sigma = np.array([[sigma_x**2,           rho*sigma_x*sigma_y],
                  [rho*sigma_x*sigma_y,   sigma_y**2]])

#===========================
#  格子点の作成
#===========================
# MATLAB の -4:0.1:4 に対応
x1 = np.arange(-4, 4.1, 0.1)  # x 軸方向の値
x2 = np.arange(-4, 4.1, 0.1)  # y 軸方向の値
X, Y = np.meshgrid(x1, x2)    # X, Y はそれぞれグリッド上の座標

#===========================
#  2変量正規分布の確率密度関数の計算
#===========================
# 各点の (x,y) 座標を格納する配列を作成
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# 各点における確率密度を計算（F の形状は (len(x2), len(x1)) となる）
F = multivariate_normal.pdf(pos, mean=mu, cov=Sigma)

#===========================
# 1) 3次元サーフェスプロット
#===========================
fig1 = plt.figure('3D Surface')
ax1 = fig1.add_subplot(111, projection='3d')
# surf: 補間表示は plot_surface 内部で行われる（edgecolor をなしに設定）
surf = ax1.plot_surface(X, Y, F, cmap='jet', edgecolor='none')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('2変量正規分布の同時確率')
fig1.colorbar(surf, shrink=0.5, aspect=5)

#===========================
# 2) 2次元等高線プロット
#===========================
fig2 = plt.figure('2D Contour')
cs = plt.contour(X, Y, F, levels=20, cmap='jet')  # 等高線 20 本指定
plt.xlabel('x')
plt.ylabel('y')
plt.title('2変量正規分布の等高線プロット')
plt.axis('equal')   # スケールを等倍にする
plt.colorbar(cs)
  
#===========================
# 条件付期待値のプロット
#===========================
# サンプル生成（乱数のシードを固定）
np.random.seed(1)
num_samples = 500
xy_samples = np.random.multivariate_normal(mu, Sigma, num_samples)

# 条件付期待値の式： mu_x_given_y = mu_x + rho*(sigma_x/sigma_y)*(y - mu_y)
# MATLAB では meshgrid により y(:,1) を利用しているので、Python では Y の1列目（Y[:, 0]）を利用
y_for_curve = Y[:, 0]  # 1列目：各行ごとに一定の y 値（x2 の値）
mu_x_given_y = mu_x + rho * (sigma_x / sigma_y) * (y_for_curve - mu_y)

# プロット
fig3, ax3 = plt.subplots()
# 等高線プロット
cs3 = ax3.contour(X, Y, F, levels=20, cmap='jet')
plt.colorbar(cs3, ax=ax3)
ax3.axis('equal')
# サンプルの散布図（マーカーサイズ 10, 青、透明度 0.4）
ax3.scatter(xy_samples[:, 0], xy_samples[:, 1], s=10, color='b', alpha=0.4, label='Samples')
# 条件付期待値の曲線（赤、線幅2）
ax3.plot(mu_x_given_y, y_for_curve, 'r', linewidth=2, label='E[x|y]')

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Joint PDF, Simulated Data, and Conditional Expectation')
ax3.grid(True)
ax3.legend(loc='upper left')
ax3.set_xlim([-4, 4])
ax3.set_ylim([-4, 4])

plt.show()
