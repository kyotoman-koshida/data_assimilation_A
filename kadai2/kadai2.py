import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import csv
import os
import runge_kutta4
import mplcursors

#Xのデータの入っているout.csvまでのパス
path = '../out.csv'
# プロットするデータのinput
D = np.loadtxt(path, delimiter=',', dtype='float32')
#ルンゲクッタの計算ステップ回数
N = 20

#errorは各Xとその時間発展事の誤差の大きさを集めるリスト
#176=(900-20)/5
error1 = [[] for i in range(176)]#微小時間h=0.01のときのため
error2 = [[] for i in range(176)]#微小時間h=0.02のときのため
error3 = [[] for i in range(176)]#微小時間h=0.03のときのため
error4 = [[] for i in range(176)]#微小時間h=0.04のときのため

#range(20,5,1000)だと何回目の計算か分かりにくいので別にcountを定義してカウントする
count = 0

#ホフメラー図を確認したところ、20番目のXはすでにカオスに移行していたので20番目から５飛ばしで1000まで行う
for i in range(20, 900, 5):

    # 誤差発達率を導出するための誤差を複数の40次元分用意する
    R = [[] for k in range(100)]
    for j in range(100):
        for k in range(40):
            R[j].append(np.random.randn()/10000)#これで40次元の誤差が100通りできた

    #Xのある点の各要素に誤差を加える
    DD = [[] for k in range(100)]#DDの要素はリストで、さらにそのリストは同一の点の100通りの誤差が入る
    for j in range(40):
        for k in range(100):
            DD[j].append( D[i][j] + R[k][j])#ある一つの点に100通りの誤差を加える

    #上で誤差を加えたXがルンゲクッタの時間発展とともにどれだけ誤差を大きくしていくかをみる
    for k in range(100):
        DDD1[k] = runge_kutta4.Lorenz96_RK4(DD[k], 0.01, N, 8.0)#微小時間h=0.01のときのため
        DDD2[k] = runge_kutta4.Lorenz96_RK4(DD[k], 0.02, N, 8.0)#微小時間h=0.02のときのため
        DDD3[k] = runge_kutta4.Lorenz96_RK4(DD[k], 0.03, N, 8.0)#微小時間h=0.03のときのため
        DDD4[k] = runge_kutta4.Lorenz96_RK4(DD[k], 0.04, N, 8.0)#微小時間h=0.04のときのため

    #ルンゲクッタでの時間発展の各ステップ段階での誤差をerrorリストに代入する
    for j in range(N):
        #deltaDはあるXの各要素の誤差を集めたリスト
        deltaD1 = [[] for k in range(176)]#微小時間h=0.01のときのため
        deltaD2 = [[] for k in range(176)]#微小時間h=0.02のときのため
        deltaD3 = [[] for k in range(176)]#微小時間h=0.03のときのため
        deltaD4 = [[] for k in range(176)]#微小時間h=0.04のときのため

        for k in range(100):
            deltaD1[k].append( D[i + (j+1)] - DDD1[k][j] ) #D[i + (j+1)]は、out.csvにある40次元Xの各要素のタイムステップあたりの真の値で、DDD[j]は誤差を含めて開始した各微小時間hごとに計算した値
            deltaD2[k].append( D[i + 2*(j+1)-1] - DDD2[k][j] )#以下jを二倍三倍四倍しているのは、元データのout.csvのタイムステップが0.01であり、それに合わせて時間の刻み幅を考える必要があるから
            deltaD3[k].append( D[i + 3*(j+1)-2] - DDD3[k][j] )
            deltaD4[k].append( D[i + 4*(j+1)-3] - DDD4[k][j] )
        #誤差の大きさ（ノルム）
    for j in range(176):
        for k in range(100):
            error1[j][k].append(np.linalg.norm(deltaD1[k]))#微小時間h=0.01のときのため
            error2[j][k].append(np.linalg.norm(deltaD2[k]))#微小時間h=0.02のときのため
            error3[j][k].append(np.linalg.norm(deltaD3[k]))#微小時間h=0.03のときのため
            error4[j][k].append(np.linalg.norm(deltaD4[k]))#微小時間h=0.04のときのため
        
    count+=1

#AERは平均誤差発達率
AER1 = []
AER2 = []
AER3 = []
AER4 = []
#erは各タイムステップごとの、アトラクタ上のある一点の100通りの誤差の平均を、さらにアトラクタ上の他の147通りで平均を取ったもののリスト
er1 = []#h=0.01のとき
er2 = []#h=0.02のとき
er3 = []#h=0.03のとき
er4 = []#h=0.04のとき
#各微小時間ごとの、さらに時間ステップごとの誤差をリストに付け加えていく
for i in range(N):
    for j in range(176):
        er1.append(np.mean(error1[j][k] for k in range(100)))
        er2.append(np.mean(error2[j][k] for k in range(100)))
        er3.append(np.mean(error3[j][k] for k in range(100)))
        er4.append(np.mean(error4[j][k] for k in range(100)))
    AER1.append(np.mean(er1))
    AER2.append(np.mean(er2))
    AER3.append(np.mean(er3))
    AER4.append(np.mean(er4))
    """
    AER1.append(np.mean([error1[k][i] for k in range(176)]))#kで繰り返しを行うことは、アトラクター上から採ってきたサンプルごとに考えているということ
    AER2.append(np.mean([error2[k][i] for k in range(176)]))#そして、error[k][i]をすべてのkで足し合わせることは、サンプルから出てきたタイムステップiのときの誤差を足し合わせることとなる
    AER3.append(np.mean([error3[k][i] for k in range(176)]))
    AER4.append(np.mean([error4[k][i] for k in range(176)]))
    """

#各微小時間ごとの平均誤差発達率をプロットする
fig = plt.figure( figsize=(11, 5) )
ax1 = fig.add_subplot(121)
ax1.plot(AER1,label="h=0.01")
ax1.plot(AER2,label="h=0.02")
ax1.plot(AER3,label="h=0.03")
ax1.plot(AER4,label="h=0.04")
ax1.set_xlabel("タイムステップ数",fontname="MS Gothic")
ax1.set_ylabel("平均誤差発達率",fontname="MS Gothic")
ax1.set_xticks( np.arange(0, N+1, 2))
ax1.legend()
ax1.set_title("微小時間hごとの平均誤差発達率",fontname="MS Gothic")

ax2 = fig.add_subplot(122)
ax2.plot(AER1,label="h=0.01")
ax2.set_xlabel("タイムステップ数",fontname="MS Gothic")
ax2.set_ylabel("平均誤差発達率",fontname="MS Gothic")
ax2.set_xticks( np.arange(0, N+1, 2))
ax2.axhline(y=2, color='gray', ls='--')
ax2.legend()
ax2.set_title("微小時間hごとの平均誤差発達率",fontname="MS Gothic")
#lines = ax2.plot(ax2,'s-')
#mplcursors.cursor(lines)

plt.show()

#nperror1 = np.array(error1)
#print(len(nperror1[:,0]))

print(D[-1])
print(len(D[1]))
#print(len(DDD1[1]))
