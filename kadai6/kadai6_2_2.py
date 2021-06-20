#   author: Tan Moriteru

import numpy as np
rm = np.random
import time
bgn = time.time()
import csv
path1 = '../observe.csv'
import runge_kutta4

J=40
F=8
h=0.01  # 予報サイクル：6hおき
dt=0.01            # 計算の単位時間
a=float(input())
L = list(range(J))


D = np.loadtxt(path1, delimiter=',', dtype='float32')       # 観測値
#D_t = np.loadtxt(path +'truth.csv', delimiter=',', dtype='float32')     # 真値
yo = D#観測値を用意する
B = np.identity(J)
total_step = len(D)

def Enorm(x, t):           # Root Mean Square Error : root(sum 1~n ai^2)/n
    x = np.reshape(np.array(x), J)
    x -= t
    n = len(x)
    s = 0
    while n > 0:
        n -= 1
        s += x[n]**2
    return np.sqrt(s/len(x)) 

"""
f = lambda x: np.matrix([ (x[0,(i+1)%J] -x[0,i-2]) *x[0,i-1] -x[0,i] +F for i in range(J)])
def model(x):           # 1 式              model: X(t) -> X(t+h), 4次Runge-Kutta法で計算する
    for i in range( int(h/dt) ):
        k1 = dt*f(x)
        k2 = dt*f(x +0.5*k1)
        k3 = dt*f(x +0.5*k2)
        k4 = dt*f(x +k3)
        x += (k1 +2*k2 +2*k3 +k4)/6.0
    return x
"""

def err_cov(n):         # 観測データ -> 誤差共分散行列
    # 観測データ誤差は正規分布乱数なので <e*e.T> = I_40
    return np.identity(n)

def obs_fix(n):         # 常に 1~n の n 点を観測データとして扱う
    H = np.block( [np.identity(n), np.zeros((n, J-n))] )
    return np.matrix(H)

def obs_rm(n):          # n点ランダムに選んで観測する
    O = rm.choice(J, size=n, replace=False)
    O.sort()
    H = np.zeros((n, J))
    for i in range(n):
        H[i, O[i]] = 1
    return np.matrix(H)

T=360*4
for j in range(5):
    L = [f'a={a}'] + [100]*(J+1)
    for n in range(J+1):
        e = 0
        xa = np.matrix(yo[rm.randint(J)])  
        for i in range(T):
            sol = runge_kutta4.Lorenz96_RK4(xa,h,5,8.0)
            xf = np.array(sol[5])#このようにまわりくどいことをしないとエラーになる（Lorenz96_rk4の戻り値の問題）
            
            H = obs_rm(n)
            y = H * np.matrix(y[i+1]).T
            K = B*H.T *np.linalg.inv( H*B*H.T + err_cov(n) )
            xa = xf + ( K*(y- H*xf.T) ).T
            if i > 40*4:
                e += (Enorm(xa, y[i+1]) - e)/ (i-40*4)
        if e < L[n+1]:
            L[n+1] = e
        if n%20 == 0:    
            cur = time.time()
            print(f'loop {j+1}, n={n} finished at {cur-bgn: .3f} sec.')
    
"""
with open(path+'a-obs.csv', 'a', newline='') as f:
    csv.writer(f).writerow(L)

print('process finished')
"""