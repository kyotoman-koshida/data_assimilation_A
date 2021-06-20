#Lorentz-96を４次のルンゲクッタ法を用いて実装する。パラメータ値Fをいろいろ変え、F=8の時にカオスとなることを確認する。余裕があれば、他の時間積分スキームも実装してみる。

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import csv
import os

J = 40
F = 8
dt = 0.01

#解きたい微分方程式
def diff_equ1(preprex, prex, x, nextx, F):
    return (nextx - preprex)*prex - x + F


###下準備###

#J=40個の変数の初期値0をいれる
X = F * np.ones(J)
X[19] += 0.008  #X[19]だけ初期値を上書きする

#各Xの値に対応づけるための番号のリストを作る
index = []
for i in range(J):
    index.append(i)

#計算の繰り返しをカウントする
count = 0

#実行するたびにXの値たちを書き換えたいので、前回以前の実行結果のファイルを削除しておく
os.remove('../out.csv')

###以下がメインパート###

#1000回だけ四次のルンゲクッタを繰り返す
while count < 5000:
    #ファイルに計算結果を追記で記入するために開いておく
    with open('../out.csv','ab') as f:

        #ルンゲクッタのk1リスト
        K1 = []
        for i in range(J):
            if i - 1 == -1:
                k1 = diff_equ1(X[J-2],X[J-1],X[i],X[i+1],F) * dt
            elif i -1 == 0:
                k1 = diff_equ1(X[J-1],X[i-1],X[i],X[i+1],F) * dt
            elif i+1 == J:
                k1 = diff_equ1(X[i-2],X[i-1],X[i],X[0],F) * dt
            else:
                k1 = diff_equ1(X[i-2],X[i-1],X[i],X[i+1],F) * dt
            K1.append(k1)
            
        #ルンゲクッタのk2リスト
        K2 = []
        for i in range(J):
            if i - 1 == -1:
                k2 = diff_equ1(X[J-2] + K1[J-2] / 2 ,X[J-1] + K1[J-1] / 2, X[i] + K1[i] / 2, X[i+1] + K1[i+1] / 2, F) * dt
            elif i -1 == 0:
                k2 = diff_equ1(X[J-1] + K1[J-1] / 2, X[i-1] + K1[i-1] / 2, X[i] + K1[i] / 2, X[i+1] + K1[i+1] / 2,F) * dt
            elif i+1 == J:
                k2 = diff_equ1(X[i-2] + K1[i-2] / 2, X[i-1] + K1[i-1] / 2, X[i] + K1[i] / 2, X[0] + K1[0] / 2, F) * dt
            else:
                k2 = diff_equ1(X[i-2] + K1[i-2] / 2, X[i-1] + K1[i-1] / 2, X[i] + K1[i] / 2, X[i+1] + K1[i+1] / 2, F) * dt
            K2.append(k2)

        #ルンゲクッタのk3リスト
        K3 = []
        for i in range(J):
            if i - 1 == -1:
                k3 = diff_equ1(X[J-2] + K2[J-2] / 2 ,X[J-1] + K2[J-1] / 2, X[i] + K2[i] / 2, X[i+1] + K2[i+1] / 2, F) * dt
            elif i -1 == 0:
                k3 = diff_equ1(X[J-1] + K2[J-1] / 2, X[i-1] + K2[i-1] / 2, X[i] + K2[i] / 2, X[i+1] + K2[i+1] / 2,F) * dt
            elif i+1 == J:
                k3 = diff_equ1(X[i-2] + K2[i-2] / 2, X[i-1] + K2[i-1] / 2, X[i] + K2[i] / 2, X[0] + K2[0] / 2, F) * dt
            else:
                k3 = diff_equ1(X[i-2] + K2[i-2] / 2, X[i-1] + K2[i-1] / 2, X[i] + K2[i] / 2, X[i+1] + K2[i+1] / 2, F) * dt
            K3.append(k3)

        #ルンゲクッタのk4リスト
        K4 = []
        for i in range(J):
            if i - 1 == -1:
                k4 = diff_equ1(X[J-2] + K3[J-2] ,X[J-1] + K3[J-1] ,X[i] + K3[i] ,X[i+1] + K3[i+1] ,F) * dt
            elif i -1 == 0:
                k4 = diff_equ1(X[J-1] + K3[J-1] ,X[i-1] + K3[i-1] ,X[i] + K3[i] ,X[i+1] + K3[i+1] ,F) * dt
            elif i+1 == J:
                k4 = diff_equ1(X[i-2] + K3[i-2] ,X[i-1] + K3[i-1] ,X[i] + K3[i] ,X[0] + K3[0] ,F) * dt
            else:
                k4 = diff_equ1(X[i-2] + K3[i-2] ,X[i-1] + K3[i-1] ,X[i] + K3[i], X[i+1] + K3[i+1], F) * dt
            K4.append(k1)



        #四次のルンゲクッタ法で次のステップの各Xの値を算出する
        for i in range(J):

            if i == 0:
                #新しい各Xの値を一時保管するための避難場所のリストをX2とおく
                X2 = np.zeros(J)

            #Xが循環的であることと、リストの符号が0番から始まっているために紙面のインデックスとずれることに注意しながら次の段階のXの値をゲットする
            if i < J:
                X2[i] = X[i] + (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i])/6
            

            #避難させていた各Xの値をもとのリストに戻す
            if i == J-1:
                X = copy.copy(X2)
                #ファイルに書き込み
                XX = np.array(X).reshape(1,40)#こうやって整形しないとsavetxtで各要素ごとに改行される
                np.savetxt(f, XX, delimiter=",")
                
                count += 1
                
                #描画する
                plt.plot(index,X)
                plt.show(block=False)
                input('press <ENTER> to continue')
                """
                """
                #いちいち毎回描画するのはうるさいので100回ごとに描画する
                #if count % 100 == 0:
                    
                #描画する
                plt.plot(index,X)
                plt.show(block=False)
                print(count)
                input('press <ENTER> to continue')
                
            

