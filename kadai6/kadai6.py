#三次元変分法のコード

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
path = '../observe.csv'
#課題４で保存されたファイルの行数を格納する
total_step = len(pd.read_csv(path, sep = ","))


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
    
    #pの初期値とH,Rは単位行列とする
    H = np.identity(N)
    R = np.identity(N)
    B = np.identity(N)

    #xaの初期値とyの初期値はともに課題４で保存したデータの最初の行とする
    D = np.loadtxt(path, delimiter=',')
    xa = D[0]#解析値xaの初期値を用意する
    yo = D#観測値を用意する

    XAs = []#解析値を記録していく
    XFs = []#予報値を記録していく
    Ys = []#観測値を記録していく
    #for_rmse = []#rmse算出のためのリスト
    RMSEs = []#rmseの入るリスト
    t_step = 0.25*np.array(range(total_step)) + 365#プロットするときに必要となる時間軸を用意する

    for i in range(total_step):
        ##以下の一回りの部分を繰り返す
        xf = OI1(xa)#解析ベクトルxから予報ベクトルxを取得する
        XFs.append(xf[19])#20番目だけを追加する（40次元は図に表すのがムズイから）
        K = OI2(B,H,R)#定数行列Bと観測演算子Hと誤差行列RからカルマンゲインKをゲットする
        xa = OI3(xf,yo[i],K,H)#予報ベクトルxfと観測ベクトルyoとカルマンゲインKと観測演算子Hから解析ベクトルxaをゲットする
        XAs.append(xa[19])#20番目だけを追加する（40次元は図に表すのがムズイから）
        Ys.append(yo[i][19])#０番目だけを追加する（40次元は図に表すのがムズイから）
        for_rmse =  (xa-yo[i]) ** 2 #rmse算出の準備
        rmse = np.sqrt( np.mean( for_rmse ) )
        RMSEs.append(rmse)

    XAs = np.array(XAs)
    XFs = np.array(XFs)
    Ys = np.array(Ys)

    ##以下はプロットパート

    #fig = plt.figure()

    fig, ax = plt.subplots(1, 2, figsize=(7,7))

    #全区間を描写するとき用
    #ax[0].plot(t_step, XAs, label="analysis")#解析値ベクトルの２０番目の時間発展をプロットする
    #ax[0].plot(t_step, XFs, label="forecast")#予報値ベクトルの２０番目の時間発展をプロットする
    #ax[0].plot(t_step, Ys, label="observation")#観測値ベクトルの２０番目の時間発展をプロットする

    #部分区間を描写するとき用
    ax[0].plot(t_step[:300], XAs[:300], label="analysis")#解析値ベクトルの２０番目の時間発展をプロットする
    ax[0].plot(t_step[:300], XFs[:300], label="forecast")#予報値ベクトルの２０番目の時間発展をプロットする
    ax[0].plot(t_step[:300], Ys[:300], label="observation")#観測値ベクトルの２０番目の時間発展をプロットする

    ax[0].set_xlabel("day")
    ax[0].set_ylabel("the value of x_19")
    ax[0].set_title("OI",fontname="MS Gothic")

    #ax2 = fig.add_subplot(1, 2, 2)
    ax[1].plot(t_step,RMSEs)
    ax[1].set_xlabel("day")
    ax[1].set_ylabel("RMSE")

  
    ax[0].legend()
    plt.show()#fig.show()だとすぐ消える 
    """

    ##以下はプロットパート

    #全区間を描写するとき用
    #plt.plot(t_step, XA, label="analysis")#解析値ベクトルの２０番目の時間発展をプロットする
    #plt.plot(t_step, XF, label="forecast")#予報値ベクトルの２０番目の時間発展をプロットする
    #plt.plot(t_step, yy, label="observation")#観測値ベクトルの２０番目の時間発展をプロットする

    #部分区間を描写するとき用
    plt.plot(t_step[:300], XA[:300], label="analysis")#解析値ベクトルの２０番目の時間発展をプロットする
    plt.plot(t_step[:300], XF[:300], label="forecast")#予報値ベクトルの２０番目の時間発展をプロットする
    plt.plot(t_step[:300], yy[:300], label="observation")#観測値ベクトルの２０番目の時間発展をプロットする

    plt.xlabel("day")
    plt.ylabel("the value of x_19")
    plt.title("OI",fontname="MS Gothic")

    plt.legend()
    plt.show()
    """

if __name__ == "__main__":
    main()