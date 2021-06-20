import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import runge_kutta4


#Xのデータの入っているout.csvまでのパス
path = '../out.csv'
# プロットするデータのinput
D = np.loadtxt(path, delimiter=',', dtype='float32')
#ルンゲクッタの計算ステップ回数
N = 200
#アトラクタから採ってくるサンプルの数
S = 100

#errorは各Xとその時間発展事の誤差の大きさを集めるリスト
error1 = [[] for i in range(S)]#微小時間h=0.01のときのため
error2 = [[] for i in range(S)]#微小時間h=0.02のときのため
error3 = [[] for i in range(S)]#微小時間h=0.03のときのため
error4 = [[] for i in range(S)]#微小時間h=0.04のときのため

#range(20,5,1000)だと何回目の計算か分かりにくいので別にcountを定義してカウントする
count = 0

#ホフメラー図を確認したところ、20番目のXはすでにカオスに移行していたので20番目から５飛ばしで1000まで行う
arr = np.random.randint(200, 4000, S)  # 20以上4000以下の整数からS個ランダムに整数を取ってくる
Arr = np.sort(arr)

# 誤差発達率を導出するための誤差を40次元分用意する
R = []
for j in range(40):
    R.append(np.random.randn())

for i in Arr:#iは500個ある

    """
    # 誤差発達率を導出するための誤差を40次元分用意する
    R = []
    for j in range(40):
        R.append(np.random.randn()/10)
    """

    #Xのある点の各要素に誤差を加える
    DD = []
    for j in range(40):
        DD.append( D[i][j] + R[j])

    #上で誤差を加えたXがルンゲクッタの時間発展とともにどれだけ誤差を大きくしていくかをみる
    DDD1 = runge_kutta4.Lorenz96_RK4(DD, 0.01, N, 8.0)#微小時間h=0.01のときのため #ルンゲクッタの計算をN回繰り返した結果をまとめたリストで、サイズがN+1のリストで、その要素が40次元のリストである入れ子構造
    DDD2 = runge_kutta4.Lorenz96_RK4(DD, 0.02, N, 8.0)#微小時間h=0.02のときのため
    DDD3 = runge_kutta4.Lorenz96_RK4(DD, 0.03, N, 8.0)#微小時間h=0.03のときのため
    DDD4 = runge_kutta4.Lorenz96_RK4(DD, 0.04, N, 8.0)#微小時間h=0.04のときのため

    #ルンゲクッタでの時間発展の各ステップ段階での誤差をerrorリストに代入する
    for j in range(N):
        #deltaDはあるXの各要素の誤差を集めたリスト
        deltaD1 = []#微小時間h=0.01のときのため
        deltaD2 = []#微小時間h=0.02のときのため
        deltaD3 = []#微小時間h=0.03のときのため
        deltaD4 = []#微小時間h=0.04のときのため

        deltaD1 = D[i + (j + 1)] - DDD1[j] #D[i + (j+1)]は、out.csvにある40次元Xの各要素のタイムステップあたりの真の値で、DDD[j]は誤差を含めて開始した各微小時間hごとに計算した値
        deltaD2 = D[i + 2*(j+1)-1] - DDD2[j]#以下jを二倍三倍四倍しているのは、元データのout.csvのタイムステップが0.01であり、それに合わせて時間の刻み幅を考える必要があるから
        deltaD3 = D[i + 3*(j+1)-2] - DDD3[j]
        deltaD4 = D[i + 4*(j+1)-3] - DDD4[j]
        #誤差の大きさ（ノルム）
        error1[count].append(np.linalg.norm(deltaD1))#微小時間h=0.01のときのため
        error2[count].append(np.linalg.norm(deltaD2))#微小時間h=0.02のときのため
        error3[count].append(np.linalg.norm(deltaD3))#微小時間h=0.03のときのため
        error4[count].append(np.linalg.norm(deltaD4))#微小時間h=0.04のときのため
        
    count+=1

#AERは平均誤差発達率
AER1 = [1.0]
AER2 = [1.0]
AER3 = [1.0]
AER4 = [1.0]


"""
R = [[] for j in range(S)]
for i in range(S):#iは500個ある

    # 誤差発達率を導出するための誤差を40次元分用意する
    for j in range(40):
        R[i].append(np.random.randn())

print(np.size(R))
"""


#各微小時間ごとの、さらに時間ステップごとの誤差をリストに付け加えていく
for i in range(N):
    AER1.append(np.mean([error1[k][i]/error1[k][0] for k in range(S)]))#S個のサンプリング点についての誤差発達率を求める
    AER2.append(np.mean([error2[k][i]/error2[k][0] for k in range(S)]))
    AER3.append(np.mean([error3[k][i]/error3[k][0] for k in range(S)]))
    AER4.append(np.mean([error4[k][i]/error4[k][0] for k in range(S)]))
    

#各微小時間ごとの平均誤差発達率をプロットする
fig = plt.figure( figsize=(11, 5) )
ax1 = fig.add_subplot(121)
ax1.plot(AER1,label="h=0.01")
ax1.plot(AER2,label="h=0.02")
ax1.plot(AER3,label="h=0.03")
ax1.plot(AER4,label="h=0.04")
ax1.set_xlabel("タイムステップ数",fontname="MS Gothic")
ax1.set_ylabel("平均誤差発達率",fontname="MS Gothic")
ax1.set_xticks( np.arange(0, N, 10))
ax1.legend()
ax1.set_title("微小時間hごとの平均誤差発達率",fontname="MS Gothic")

ax2 = fig.add_subplot(122)
ax2.plot(AER1,label="h=0.01")
ax2.set_xlabel("タイムステップ数",fontname="MS Gothic")
ax2.set_ylabel("平均誤差発達率",fontname="MS Gothic")
ax2.set_xticks( np.arange(0, N, 10))
ax2.axhline(y=2, color='gray', ls='--')
ax2.legend()
ax2.set_title("微小時間hごとの平均誤差発達率",fontname="MS Gothic")
#lines = ax2.plot(ax2,'s-')
#mplcursors.cursor(lines)
for i in range(S):
    print(error1[i][0])
plt.show()

#nperror1 = np.array(error1)
#print(nperror1[:,0])
#print(len(nperror1[:,0]))

