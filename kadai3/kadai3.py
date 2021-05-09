import runge_kutta4
import os
import numpy as np

#secondyear.csvは最初の一年をスピンアップとして捨てて、後半一年分を六時間ごとに保存しているもの
#実行するたびにXの値たちを書き換えたいので、前回以前の実行結果のファイルを削除しておく
os.remove('../secondyear.csv')
#14600=0.2*(365*2)/0.01 (一日=0.2)で二年分積分するのだから365×2日では365*2*0.2時間タイムステップとなる。このルンゲクッタ４では0.01タイムステップで行うから、0.01で割ることで何回計算すべきかわかる
sol = runge_kutta4.Lorenz96_RK4([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8.008, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 0.01, 14600, 8.0)
#ファイルに計算結果を追記で記入するために開いておく
with open('../secondyear.csv','ab') as f:
    for i in range(7300,14600,5):#最初の一年をスピンアップとして捨てるから7300スタート、六時間ごとに保存するために5(0.2=一日だから)刻みで保存
        XX = np.array(sol[i]).reshape(1,40)#こうやって整形しないとsavetxtで各要素ごとに改行される
        np.savetxt(f, XX, delimiter=",")

    
