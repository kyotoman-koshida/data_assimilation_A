#   author: Tan Moriteru

import numpy as np
path = 'C:/code/python/data-asmp/result/'

J=40;   F=8
dt=0.005        # 計算の単位時間
h=0.05          # 予報サイクル：6h 分
D = np.loadtxt(path +'obs.csv', delimiter=',', dtype='float32')

# L96-40 変数モデルに(形式的に) Kalman-Filter を実装する
#   Forecast:   (1)-    xf|i = m(xa|i-1)
#               (2)-    Pf|i = M Pa|i-1 M^T
#   Analysis:   (3)-    xa|i = xf|i + K|i (yo|i - H|i xf|i)
#               (4)-    Pa|i = (1- K|i H|i) Pf|i
#               (5)-    K|i  = Pf|i H|i^T (H|i Pf|i H|i^T +R|i)^-1

def f_rk4(x):           # L96を4次Runge-Kutta法で計算する
    f = lambda x: np.array([ (x[(i+1)%J] -x[i-2]) *x[i-1] -x[i] +F for i in range(J)])
    k1 = dt*f(x);           k2 = dt*f(x +0.5*k1)
    k3 = dt*f(x +0.5*k2);   k4 = dt*f(x +k3)
    return (k1 +2*k2 +2*k3 +k4)/6.0

def err_cov(x):         # 観測データ -> 誤差共分散行列
    # ここよくわかんない
    return P    # np.matrix 型にしておく

def forecast(x, P):     # 1,2 式
    # 2 式
    m = lambda i: h*[-x[i-1], x[(i+1)%J]-x[i-2], -1, x[i-1]] + [0 for i in range(J-4)]
    dM = np.matrix([ np.roll(m(i), i) for i in range(J) ]);      dM += np.identity(J)
    P = dM * P * dM.T
    # 1 式      M: X(t) -> X(t+h)
    for i in range( int(h/dt) ):
        x += f_rk4(x)
    return x, P

def analysis(x, y, P):  # 3,4,5 式      H = I_40 としてよい（？）
    K = P *((P - err_cov(y)) **-1)      # 5 式
    x += ( K*(y-x).T ).T;   P -= K * P      # 3,4 式
    return x, P

x = np.matrix(D[0]);  P = err_cov(x)
for i in range(len(D)-1):
    x, P = forecast(x, P)
    y = np.matrix(D[i+1])
    x, P = analysis(x, y, P)

