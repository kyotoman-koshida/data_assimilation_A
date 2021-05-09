import numpy as np

def Lorenz96_RK4(initial, h, N, F):
    
    #initialはfloatを要素にもつ長さ40のndarray(初期値)
    #hはfloat(微小時間)
    #Nはint(時間hNまで計算する)
    #Fはfloat(パラメータ)
    
    def G(y):

        return np.array([(y[(j+1) % 40] - y[(j-2) % 40]) * y[(j-1) % 40] - y[j] + F for j in range(0, 40)])

    #Lorenz96の右辺

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

initials, z, initials2 = [], [], []

for i in range(0,80):

    e = np.zeros(40)

    e[0] = (i+1)*0.001

    initials.append(np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]) + e)

#固定点からずらしたベクトルを80個作る。
    
for i in range(0,80):

    initials2.append(Lorenz96_RK4(initials[i], 0.05, 500, 8.0)[500])

    z.append(Lorenz96_RK4(initials2[i], 0.05, 8, 8.0)[8])

#上のベクトルから500ステップ後の値を初期値とする。zにさらに8ステップ後の値を入れる。

b = []

for j in range(0,80):

    y = []

    a = np.array([])

    for i in range(0,40):

        e = np.zeros(40)

        e[i] = 0.001

        y.append(Lorenz96_RK4(initials2[j] + e, 0.05, 8, 8.0)[8])

        y.append(Lorenz96_RK4(initials2[j] - e, 0.05, 8, 8.0)[8])

    for i in range(0,80):

        a = np.append(a, np.linalg.norm(y[i] - z[j]))

    a = a / 0.001

    b = np.append(b, np.mean(a))

c = np.mean(b)

print(c)

    

        

        

    









    
    
    
