#   author: Tan Moriteru

import numpy as np
import csv
path = 'C:/Users/heihe/Python/dtas/out.csv'   # 出力先設定

# Lorenz-96 model を 4次元Runge-Kutta法を用いて実装する
# L96model： dX_j/dt = ( X_j+1 -X_j-2 )* X_j-1 -X_j + F , 1 <= j <= N 
# ここでは周期境界条件 X_j+N = X_j を入れるとし、N=40とする

# 定数の設定
J=40
F=8.0          # 方程式第三項の定数
h=0.01         # 単位時間
M=4*10         # 計算回数: 4*(日数)

# X の初期値
X=[F for i in range(J)]
X[19] *= 1.001    
X = np.array(X)


# 時間積分scheme の実装
def d(X):       # L96 の右辺を与える関数
    return np.array([ (X[(i+1)%J] -X[i-2]) *X[i-1] -X[i] +F for i in range(J)])

fwd = lambda X: h* d(X)     # 前進差分
rk2 = lambda X: ( fwd(X) +fwd(X +fwd(X)) )/2    # Runge-Kutta 法（2次）

def rk4(X):     # Runge-Kutta 法（4次）
    k1 = fwd(X)
    k2 = fwd(X +0.5*k1)
    k3 = fwd(X +0.5*k2)
    k4 = fwd(X +k3)
    return (k1 +2*k2 +2*k3 +k4)/6.0


# 数値計算の実行
with open(path, 'w', newline='') as f:
    csv.writer(f).writerow(X)

while M > 0:
    X += rk4(X)
    M -= 1
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(X)

print('process finished')
