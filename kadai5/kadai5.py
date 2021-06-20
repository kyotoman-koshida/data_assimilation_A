#カルマンフィルターのコード

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import runge_kutta4
from sklearn.metrics import mean_squared_error

#
h = 0.01
#Nは状態変数の次元
N = 40
#
F = 8.0
#Xのデータの入っているobserve.csvまでのパス
path = '../observe.csv'
#観測誤差共分散行列の準備
R = np.identity(40)
#観測演算子
H = np.identity(40)
#ヤコビアンで使うための単位行列
I = np.identity(40)
#inflation
alpha = 1.1
#課題４で保存されたファイルの行数を格納する
total_step = len(pd.read_csv(path, sep = ","))

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

#カルマンフィルタの式の3式
def KF3(xf, y, K, H):
    xf = np.array(xf)
   
    pre = y - np.dot(H ,xf)#計算途中。(y-Hx)の部分。
    xa = xf + np.dot(K ,pre)
   
    xa = np.array(xa)#これしないとxaはnpmatrix型になっちゃう
    
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
            #pre2_1 = runge_kutta4.Lorenz96_RK4(x + delta[0]*e0, h, 5, F)
            #pre2_1 = runge_kutta4.Lorenz96_RK4(x + h*e0, h, 5, F)
            pre2_1 = runge_kutta4.Lorenz96_RK4(x + 0.000001*e0, h, 5, F)
            pre2_2 = runge_kutta4.Lorenz96_RK4(x, h, 5, F)
            #pre = (pre2_1[5]-pre2_2[5]) / np.linalg.norm(delta[0]*e0)
            #pre = (pre2_1[5]-pre2_2[5]) / np.linalg.norm(h*e0)
            pre = (pre2_1[5]-pre2_2[5]) / 0.000001
            pre = (np.array([pre]).T)#縦ベクトルに変換
        #pre2_1 = runge_kutta4.Lorenz96_RK4(x + delta[i+1]*e, h, 5, F)
        #pre2_1 = runge_kutta4.Lorenz96_RK4(x + h*e, h, 5, F)
        pre2_1 = runge_kutta4.Lorenz96_RK4(x + 0.000001*e, h, 5, F)
        pre2_2 = runge_kutta4.Lorenz96_RK4(x, h, 5, F)
        
        #pre2 = (pre2_1[5]-pre2_2[5])/ np.linalg.norm(delta[i+1]*e)#５番目を取るのは0.05(六時間)=h*5だから
        #pre2 = (pre2_1[5]-pre2_2[5])/ np.linalg.norm(h*e)
        pre2 = (pre2_1[5]-pre2_2[5])/ 0.000001
        pre2 =( np.array([pre2]).T )#縦ベクトルに変換
        pre = np.hstack([pre,pre2]) 

    #M = pre / np.linalg.norm(delta)
    M = pre
    
    return M

#以下、メインパート
def main():

    #pの初期値とH,Rは単位行列とする
    H = np.identity(N)
    R = np.identity(N)
    Pa = 8*8*np.identity(N)#アトラクタの帯域的な大きさを表すために8*8をしている

    #xaの初期値とyの初期値はともに課題４で保存したデータの最初の行とする
    D = np.loadtxt(path, delimiter=',')
    #xa = D[0]
    xa = D[0] + [ 30*np.random.normal(0,1) for i in range(N) ]#解析値xaの初期値を用意する
    #xa = D[0] + [ 30 for i in range(N) ]#解析値xaの初期値を用意する
    y = D#観測値を用意する


    #ヤコビアン旧版
    """
    #ヤコビアンMの実装
    m = lambda i: [-xa[(i-1)%N], (xa[(i+1)%N]-xa[(i-2)%N]), -1, xa[(i-1)%N]] + [0 for i in range(N-4)]#第一項でhを括り出すとエラーになる
    M = np.matrix([ np.roll(m(i), i) for i in range(N) ])
    """

    XAs = []#解析値を記録していく
    XFs = []#予報値を記録していく
    Ys = []#観測値を記録していく
    #for_rmse = []#rmse算出のためのリスト
    RMSEs = []#rmseの入るリスト
    t_step = 0.25*np.array(range(total_step)) + 365#プロットするときに必要となる時間軸を用意する

    for i in range(total_step):
        ##以下のひとかたまりの部分を繰り返す
        xf = KF1(xa)#解析ベクトルxから予報ベクトルxを取得する
        XFs.append(xf[19])#２０番目だけを追加する（40次元は図に表すのがムズイから）

        #ヤコビアン新版
        M = jacobian(xa)

        Pf = KF2(M, Pa)#解析誤差共分散行列Paから予報誤差共分散行列Pfを取得する
        K = KF5(Pf, H, R)#予報誤差共分散行列PfとHとRからカルマンゲインKを取得する
        xa = KF3(xf, y[i], K, H)#予報ベクトルxと観測値yとカルマンゲインKとHから解析ベクトルxを取得する
        XAs.append(xa[19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
        Ys.append(y[i][19])#２０番目だけを追加する（40次元は図に表すのがムズイから）
        Pa = KF4(K, H, Pf)#カルマンゲインKとHと予報誤差共分散行列Pfから解析誤差共分散行列Paを取得する

        for_rmse = (xa-y[i]) ** 2 #rmse算出の準備
        rmse = np.sqrt( np.mean( for_rmse ) )
        RMSEs.append(rmse)

        #rmse = np.sqrt(mean_squared_error(xa, y[i]))
        #RMSEs.append(rmse)

    XAs = np.array(XAs)
    XFs = np.array(XFs)
    Ys = np.array(Ys)


    ##以下はプロットパート

    #fig = plt.figure()

    fig, ax = plt.subplots(1, 2, figsize=(9,6))

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
    ax[0].set_title(f"inflationが {alpha} のとき",fontname="MS Gothic")

    #ax2 = fig.add_subplot(1, 2, 2)
    ax[1].plot(t_step,RMSEs)
    ax[1].set_xlabel("day")
    ax[1].set_ylabel("RMSE of analysis")

  
    ax[0].legend()
    plt.show()#fig.show()だとすぐ消える    

if __name__ == "__main__":
    main()