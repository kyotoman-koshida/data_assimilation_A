import numpy as np
import matplotlib.pyplot as plt
import os


#Xのデータの入っているout.csvまでのパス
path1 = '../secondyear.csv'
#Xのデータの入っているout.csvまでのパス
path2 = '../observe.csv'

# プロットするデータのinput
D = np.loadtxt(path1, delimiter=',', dtype='float32')

#errorは各Xとその時間発展事の誤差の大きさを集めるリスト
N = (14600-7300)/5
errors = []#後のヒストグラム作成のために誤差のリストを作っておく

for i in range(int(N)):
    for j in range(40):
        error = np.random.normal(0,1)
        errors.append(error) #後のヒストグラムのために誤差だけも集めておく
        D[i][j] += error #もともと保存していたデータに誤差を足しこむ

#実行するたびにXの値たちを書き換えたいので、前回以前の実行結果のファイルを削除しておく
os.remove('../observe.csv')
#ファイルに計算結果を追記で記入するために開いておく
with open('../observe.csv','ab') as f:
#ファイルに書き込み
    XX = np.array(D).reshape(int(N),40)#こうやって整形しないとsavetxtで各要素ごとに改行される
    np.savetxt(f, XX, delimiter=",")

print(len(errors))
#誤差のヒストグラムを描く
(a_hist, a_bins, _) = plt.hist(errors,bins=50,density=True)
fig = plt.figure( figsize=(11, 5) )
ax1 = fig.add_subplot(111)
ax1.set_xlabel("標準正規分布のヒストグラフ",fontname="MS Gothic")
ax1.legend()
plt.show()

