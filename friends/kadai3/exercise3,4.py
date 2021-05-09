import numpy as np

def Lorenz96_RK4(initial, h, N, F):
    
    #initialはfloatを要素にもつ長さ40のlist(初期値)
    #hはfloat(微小時間)
    #Nはint(時間hNまで計算する)
    #Fはfloat(パラメータ)
    
    def G(y):

        return np.array([(y[(j+1) % 40] - y[(j-2) % 40]) * y[(j-1) % 40] - y[j] + F for j in range(0, 40)])

    #Lorenz96の右辺
        
    initial = np.array(initial)

    #初期値の型をndarrayにする。

    list, x = [initial], initial

    #この'list'の第n成分に時間nにおける値をndarrayにまとめて入れる。'x'に初期値を代入する。

    for i in range(0,N):

        q1 = G(x) * h

        q2 = G(x + q1 / 2) * h

        q3 = G(x + q2 / 2) * h

        q4 = G(x + q3) * h

        x = x + (q1 + q2 * 2 + q3 * 2 + q4) / 6

        list.append(x)

    return list

    #第iステップで時刻hiでの値を求め、'list'のiオフセットと'x'に入れる。



data_ex3 = Lorenz96_RK4([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8.008, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 0.05, 2920, 8.0)

#2年分計算する。

del data_ex3[0:1461]

#最初の1年分を取り除く。

import copy

data_ex4 = copy.deepcopy(data_ex3)

import random

rnd_data = []

for i in range(0,1460):

    for j in range(0,40):

        rnd_data.append(random.normalvariate(0,1))

        data_ex4[i][j] += rnd_data[-1]



#各ステップでrnd_dataに乱数を入れる。入れた乱数を一年分のデータの各次元の成分に足し合わせていく。

import matplotlib.pyplot as plt

plt.hist(rnd_data)

plt.show()

#乱数をヒストグラムに表示する。







    
    
    
