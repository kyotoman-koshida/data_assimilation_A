#観測点を減らしていくカルマンフィルタのコード

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import runge_kutta4

rm = np.random
bgn = time.time()

#予報サイクル6時間
h = 0.01
#Nは状態変数の次元
N = 40
#
F = 8.0
#Xのデータの入っているobserve.csvまでのパス
path1 = '../observe.csv'
#inflation
alpha = 1.0
#ヤコビアンで使うための単位行列
I = np.identity(40)

def Enorm(x, t):#平均二乗誤差を求める関数
    x = np.reshape(np.array(x), N)
    x -= t
    n = len(x)
    s = 0
    while n > 0:
        n -= 1
        s += x[n]**2
    return np.sqrt(s/len(x)) 

def err_cov(n):         # 観測データ -> 誤差共分散行列
    # 観測データ誤差は正規分布乱数なので <e*e.T> = I_40
    return np.identity(n)

def obs_fix(n):         # 常に 1~n の n 点を観測データとして扱う
    H = np.block( [np.identity(n), np.zeros((n, N-n))] )
    return np.matrix(H)

def obs_rm(n):          # n点ランダムに選んで観測する
    O = rm.choice(N, size=n, replace=False)
    O.sort()
    H = np.zeros((n, N))
    for i in range(n):
        H[i, O[i]] = 1
    return np.matrix(H)

#カルマンフィルタの式の1式
def KF1(xa):#解析ベクトルをうける
    sol = runge_kutta4.Lorenz96_RK4(xa, h, 5, F)
    xf = np.array(sol[5])#solの2行目が所望のデータである。このようにまわりくどいことをしないとエラーになる（Lorenz96_rk4の戻り値の問題）
    return xf #予報ベクトルを返す  

#カルマンフィルタの式の2式
def KF2(M, Pa):
    pre = np.dot(M, Pa)#計算途中。MPの部分。
    Pf =alpha * np.dot(pre, M.T)#alphaをかけたのはinfrationの役割をもたせるため
    return Pf

def KF3(xf,y,K,H):

    xf =( np.array([xf]).T )#縦ベクトルに変換
    pre = y - np.dot(H,xf)#pre1は行列やベクトルの計算とかしてる
    pre2 = np.dot(K,pre).T
    xf = np.array(xf).T #横ベクトルに変換
    pre2 = np.array(pre2)
    xa = xf[0] + pre2[0]
    return xa

#カルマンフィルタの式の4式
def KF4(K, H, Pf):
    I = np.identity(N)
    pre = I - np.dot(K, H)#計算途中。(1-KH)の部分。
    Pa = np.dot(pre, Pf)
    return Pa

#カルマンフィルタの式の5式
def KF5(Pf, H, R):

    pre1 = np.dot(Pf, H.T)#計算途中。PHの部分
    pre2 = np.dot(H, pre1) + R#計算途中。(HPtH+R)の部分。
    pre3 = np.linalg.inv(pre2)#計算途中。(HPtH+R)の逆行列をとっている。
    K = np.dot(pre1, pre3)
    return K


#ヤコビアンの実装
def jacobian(x):
    #delta = np.array([np.random.normal(0,1) for i in range(N)]) * 1.0e-5
    for i in range(N-1):
        e = I[:,i+1]
        if i == 0:
            e0 = I[:,0]
            pre2_1 = runge_kutta4.Lorenz96_RK4(x + h*e0, h, 5, F)
            pre2_2 = runge_kutta4.Lorenz96_RK4(x, h, 5, F)
            pre = (pre2_1[5]-pre2_2[5]) / np.linalg.norm(h*e0)
            pre = (np.array([pre]).T)#縦ベクトルに変換
        pre2_1 = runge_kutta4.Lorenz96_RK4(x + h*e, h, 5, F)
        pre2_2 = runge_kutta4.Lorenz96_RK4(x, h, 5, F)
        pre2 = (pre2_1[5]-pre2_2[5])/ np.linalg.norm(h*e)
        pre2 =( np.array([pre2]).T )#縦ベクトルに変換
        pre = np.hstack([pre,pre2]) 

    M = pre    
    return M


#メインパート
def main():

    total_step = len(pd.read_csv(path1, sep = ","))
    #xaの初期値とyの初期値はともに課題４で保存したデータの最初の行とする
    D = np.loadtxt(path1, delimiter=',')
    
    yo = D#観測値を用意する

    Pa = 8*8*np.identity(N)#アトラクタの帯域的な大きさを表すために8*8をしている
        
    mean_RMSEs = []#観測点の数ごとのrmseのリスト
    t_step = 0.25*np.array(range(total_step)) + 365#プロットするときに必要となる時間軸を用意する


    RMSEs = [[] for i in range(N+1)]#rmseの入るリスト
    for k in range(8,N+1):
        xa = np.array(yo[rm.randint(N)])
        for i in range(120,total_step):
            ##以下の一回りの部分を繰り返す
            xf = KF1(xa)#解析ベクトルxから予報ベクトルxを取得する
            M = jacobian(xa)
            Pf = KF2(M, Pa)#解析ベクトルpから予報誤差共分散行列Pfを取得する
            H = obs_rm(k) ##移植
            R = err_cov(k)##移植
            K = KF5(Pf, H, R)#予報ベクトルpとHとRからカルマンゲインKを取得する
            y = H * np.matrix(yo[i]).T
            xa = KF3(xf, y, K, H)#予報ベクトルxと観測値yとカルマンゲインKとHから解析ベクトルxを取得する
            Pa = KF4(K, H, Pf)#カルマンゲインKとHと予報誤差共分散行列Pfから解析誤差共分散行列Paを取得する

            if i > 40*4:
                #e += (Enorm(xa, yo[i]) - e)/ (i-40*4)
                rmse = Enorm(xa, yo[i])##移植をもとに作成
                RMSEs[k].append(rmse)

    for i in range(N):
        mean_RMSEs.append(np.mean(RMSEs[i]))

    #print(mean_RMSEs)

    #実行するたびにXの値たちを書き換えたいので、前回以前の実行結果のファイルを削除しておく
    os.remove('../kf_rmse.csv')
    #ファイルに計算結果を追記で記入するために開いておく
    with open('../kf_rmse.csv','ab') as f:
        np.savetxt(f, mean_RMSEs, delimiter=",")

if __name__ == "__main__":
    main()