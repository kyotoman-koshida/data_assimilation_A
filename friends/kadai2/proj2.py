#   author: Tan Moriteru
F=8         # 方程式第三項の定数

import numpy as np;     rm = np.random 
import matplotlib.pyplot as plt
import time
path = 'C:/code/python/data-asmp/result/attractor.csv'
bgn = time.time()

# 誤差の平均発達率を調べる
# -> アトラクタ上から点をランダムにサンプリングし（一様乱数）、
#    ランダムな誤差ベクトルを加えて発達率を得る
# -> その平均をプロットする

# 定数の設定
J=40
dt=0.005         # 単位時間

# 時間積分scheme の実装
def f(X):       # L96 の右辺を与える関数
    return np.array([ (X[(i+1)%J] -X[i-2]) *X[i-1] -X[i] +F for i in range(J)])

def rk4(X):     # Runge-Kutta 法（4次）
    k1 = dt*f(X);           k2 = dt*f(X +0.5*k1)
    k3 = dt*f(X +0.5*k2);   k4 = dt*f(X +k3)
    return (k1 +2*k2 +2*k3 +k4)/6.0

D = np.loadtxt(path, delimiter=',', dtype='float32')    # アトラクタの入力
def norm(X):    # L2 ノルム
    s = 0
    for i in range(len(X)):
        s += X[i]**2
    return np.sqrt(s)

T=1           # 計算時間
def devR(X, dX):     # 発達率の時間発展を与える関数
    M = int(T/dt)
    R = [];     R.append(1)
    r = norm(dX)
    Xerr = X + dX
    while M > 0:
        X += rk4(X);    Xerr += rk4(Xerr);  M -= 1
        R.append(norm(Xerr-X)/r)
    return np.array(R)

avr = 0;       N=1000    # サンプリング数
for i in range(N):      # 平均発達率の計算
    X = D[rm.randint(0, len(D))]                       # アトラクタ上のランダムな点
    dX = rm.rand(J) *( -3+2*rm.randint(1, 3, J) )       # (-1, 1) の一様分布乱数を成分とする40次元ベクトル
    avr += devR(X, dX)     
    if i %50 == 0:
        cur = time.time()
        print(f'left: {N-i}, {cur-bgn: .3f} sec.')

cur = time.time()
print(f'process finished at {cur-bgn: .3f} sec')

# error doubling time を求める　←→ avr(t)=2 なるt を求める 
avr = avr/N;    avr -= 2
for i in range(int(T/dt)):
    if avr[i]*avr[i+1] <= 0:
        edt = ( i+ avr[i]/(avr[i]-avr[i+1]) ) *dt;      break

# プロット
fig = plt.figure( figsize=(5, 5) );     ax = fig.add_subplot()
ax.set_title(f'avr. of development rate, where F={F} and {N} samples')
ax.set_xlabel('time');      ax.set_ylabel('ratio')
ax.set_ylim(0, 5)
ax.plot( [i*dt for i in range(int(T/dt)+1) ], avr+2 )
ax.plot(edt, 2, color="green", marker="o");     ax.text(edt, 2.1, f'({edt: .4f}, 2.0)')
ax.axhline(2.0, color="gray")

plt.show()

