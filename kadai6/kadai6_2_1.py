#観測点を減らしていく三次元変分法のコード

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
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
#課題４で保存されたファイルの行数を格納する


def Enorm(x, t):           # Root Mean Square Error : root(sum 1~n ai^2)/n
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

#OIの第１式
def OI1(xa):
    sol = runge_kutta4.Lorenz96_RK4(xa,h,5,8.0)
    xf = np.array(sol[5])#このようにまわりくどいことをしないとエラーになる（Lorenz96_rk4の戻り値の問題）
    return xf

#OIの第２式
def OI2(B,H,R):
    pre1 = np.dot(B,H.T)#preは行列やベクトルの計算とかしてる
    pre2 = np.dot(H,pre1)
    pre3 = np.linalg.inv(pre2 + R)
    K = np.dot(pre1,pre3)
    return K

#OIの第３式
def OI3(xf,yo,K,H):

    xf =( np.array([xf]).T )#縦ベクトルに変換
    #print(np.shape(H@xf))
    #print(H@xf)
    pre = yo - np.dot(H,xf)#pre1は行列やベクトルの計算とかしてる
    pre2 = np.dot(K,pre).T
    
    xf = np.array(xf).T #横ベクトルに変換
    pre2 = np.array(pre2)
    xa = xf[0] + pre2[0]
    #print("xf=")
    #print(xf)
    
    #print("pre2=")
    #print(pre2[0])
    #print("xa[0]=")
    return xa


#メインパート
def main():

    total_step = len(pd.read_csv(path1, sep = ","))
    #xaの初期値とyの初期値はともに課題４で保存したデータの最初の行とする
    D = np.loadtxt(path1, delimiter=',')
    
    yo = D#観測値を用意する

    #xa = D[0]#解析値xaの初期値を用意する
    

    H = np.identity(N)

    #R = np.identity(N)
    

    B = np.identity(N)

    

    for_rmse_s = []#rmse算出のためのリスト
    
    mean_RMSEs = []#観測点の数ごとのrmseのリスト
    t_step = 0.25*np.array(range(total_step)) + 365#プロットするときに必要となる時間軸を用意する





    """
    #k番目ごとにXを観測する
    for k in range(1,N):
        RMSEs = []#rmseの入るリスト
        for i in range(120, total_step):
            ##以下の一回りの部分を繰り返す
            xf = OI1(xa)#解析ベクトルxから予報ベクトルxを取得する
            XFs.append(xf[19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
            K = OI2(B,H,R)#定数行列Bと観測演算子Hと誤差行列RからカルマンゲインKをゲットする
            xa = OI3(xf,yo[i],K,H)#予報ベクトルxfと観測ベクトルyoとカルマンゲインKと観測演算子Hから解析ベクトルxaをゲットする
            XAs.append(xa[19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
            Ys.append(yo[i][19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
            for j in range(0,N,N-k):
                for_rmse_s.append(  (yo[i][j]-xa[j]) ** 2 ) #rmse算出の準備
            rmse = np.sqrt( np.mean( for_rmse_s ) ) 
            RMSEs.append(rmse)
        mean_RMSEs.append(np.mean(RMSEs))
    
    """
    """
    #ランダムにXの観測する要素を選ぶ
    for k in range(1,N):
        RMSEs = []#rmseの入るリスト
        for i in range(120, total_step):
            ##以下の一回りの部分を繰り返す
            xf = OI1(xa)#解析ベクトルxから予報ベクトルxを取得する
            XFs.append(xf[19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
            K = OI2(B,H,R)#定数行列Bと観測演算子Hと誤差行列RからカルマンゲインKをゲットする
            xa = OI3(xf,yo[i],K,H)#予報ベクトルxfと観測ベクトルyoとカルマンゲインKと観測演算子Hから解析ベクトルxaをゲットする
            XAs.append(xa[19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
            Ys.append(yo[i][19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
            arr = np.random.randint(0, N, k)  # 0以上N以下の整数からS個ランダムに整数を取ってくる
            Arr = np.sort(arr)
            for j in Arr:
                for_rmse_s.append(  (xa[j]-yo[i][j]) ** 2 ) #rmse算出の準備
            rmse = np.sqrt( np.mean( for_rmse_s ) ) 
            RMSEs.append(rmse)
        mean_RMSEs.append(np.mean(RMSEs))
    """

    """

    RMSEs = []#rmseの入るリスト
    for i in range(120, total_step):
        ##以下の一回りの部分を繰り返す
        xf = OI1(xa)#解析ベクトルxから予報ベクトルxを取得する
        XFs.append(xf[19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
        K = OI2(B,H,R)#定数行列Bと観測演算子Hと誤差行列RからカルマンゲインKをゲットする
        xa = OI3(xf,yo[i],K,H)#予報ベクトルxfと観測ベクトルyoとカルマンゲインKと観測演算子Hから解析ベクトルxaをゲットする
        XAs.append(xa[19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
        Ys.append(yo[i][19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
        for j in range(0,1):
            for_rmse_s.append(  (yo[i][j]-xa[j]) ** 2 ) #rmse算出の準備
        rmse = np.sqrt( np.mean( for_rmse_s ) ) 
        RMSEs.append(rmse)
    mean_RMSEs.append(np.mean(RMSEs))

    print(mean_RMSEs)
    """

    RMSEs = [[] for i in range(N+1)]#rmseの入るリスト
    for k in range(N+1):
        xa = np.array(yo[rm.randint(N)])
        e = 0
        for i in range(120,total_step):
            ##以下の一回りの部分を繰り返す
            #print(np.shape(xa))
            
            xf = OI1(xa)#解析ベクトルxから予報ベクトルxを取得する
            H = obs_rm(k) ##移植
            R = err_cov(k)##移植
            K = OI2(B,H,R)#定数行列Bと観測演算子Hと誤差行列RからカルマンゲインKをゲットする
            y = H * np.matrix(yo[i]).T
            #print(np.matrix(yo[i]).T)
            #print(y)
            xa = OI3(xf,y,K,H)#予報ベクトルxfと観測ベクトルyoとカルマンゲインKと観測演算子Hから解析ベクトルxaをゲットする
            
            #arr = np.random.randint(0, N, 40)  # 0以上N以下の整数からS個ランダムに整数を取ってくる

            """
            Arr = np.sort(arr)
            for_rmse_s = []
            for j in Arr:
                for_rmse_s.append(  (xf[j]-yo[i][j]) ** 2 ) #rmse算出の準備
            
            for_rmse =  (xa-yo[i]) ** 2 #rmse算出の準備
            rmse = np.sqrt( np.mean( for_rmse ) )
            """
            if i > 40*4:
                #e += (Enorm(xa, yo[i]) - e)/ (i-40*4)
                rmse = Enorm(xa, yo[i])##移植をもとに作成
                RMSEs[k].append(rmse)


            #rmse = Enorm(xa, y)##移植をもとに作成
            #RMSEs.append(rmse)

    #mean_RMSEs = np.mean(RMSEs)

    for i in range(N):
        mean_RMSEs.append(np.mean(RMSEs[i]))

    #print(np.mean(RMSEs[0]))
    #print(np.mean(RMSEs[5]))
    #print(np.mean(RMSEs[10]))
    #print(np.mean(RMSEs[15]))
    #print(np.mean(RMSEs[20]))
    #print(mean_RMSEs)

    #実行するたびにXの値たちを書き換えたいので、前回以前の実行結果のファイルを削除しておく
    os.remove('../oi_rmse.csv')
    #ファイルに計算結果を追記で記入するために開いておく
    with open('../oi_rmse.csv','ab') as f:
        np.savetxt(f, mean_RMSEs, delimiter=",")


    """
    ##以下はプロットパート
    plt.title("観測点を減らしていくときのOIのRMSE",fontname="MS Gothic")
    plt.plot(range(1,N), mean_RMSEs, label="ランダムに観測")
    plt.show()
    """

if __name__ == "__main__":
    main()