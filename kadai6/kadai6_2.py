#観測点を減らしていく三次元変分法のコード

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import runge_kutta4

#予報サイクル6時間
h = 0.01
#Nは状態変数の次元
N = 40
#
F = 8.0
#Xのデータの入っているobserve.csvまでのパス
path1 = '../observe.csv'
#課題４で保存されたファイルの行数を格納する
total_step = len(pd.read_csv(path1, sep = ","))


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
    pre = yo - H @ xf#pre1は行列やベクトルの計算とかしてる
    xa = xf + K @ pre
    return xa


#メインパート
def main():

    H = np.identity(N)    
    R = np.identity(N)
    B = np.identity(N)

    #xaの初期値とyの初期値はともに課題４で保存したデータの最初の行とする
    D = np.loadtxt(path1, delimiter=',')
    xa = D[0]#解析値xaの初期値を用意する
    yo = D#観測値を用意する

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

    RMSEs = []#rmseの入るリスト
    for i in range(120,total_step):
        ##以下の一回りの部分を繰り返す
        xf = OI1(xa)#解析ベクトルxから予報ベクトルxを取得する
        K = OI2(B,H,R)#定数行列Bと観測演算子Hと誤差行列RからカルマンゲインKをゲットする

        xa = OI3(xf,yo[i],K,H)#予報ベクトルxfと観測ベクトルyoとカルマンゲインKと観測演算子Hから解析ベクトルxaをゲットする
        
        arr = np.random.randint(0, N,40)  # 0以上N以下の整数からS個ランダムに整数を取ってくる
        Arr = np.sort(arr)
        for_rmse_s = []
        for j in Arr:
            for_rmse_s.append(  (xf[j]-yo[i][j]) ** 2 ) #rmse算出の準備
        
        for_rmse =  (xa-yo[i]) ** 2 #rmse算出の準備
        rmse = np.sqrt( np.mean( for_rmse ) )
        RMSEs.append(rmse)

    mean_RMSEs = np.mean(RMSEs)
    print(mean_RMSEs)

    """
    ##以下はプロットパート
    plt.title("観測点を減らしていくときのOIのRMSE",fontname="MS Gothic")
    plt.plot(range(1,N), mean_RMSEs, label="ランダムに観測")
    plt.show()
    """

if __name__ == "__main__":
    main()