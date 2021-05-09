import os
import numpy as np

def Lorenz96_RK4(initial, h, N, F):
    
    #initialはfloatを要素にもつ長さ40のlist(初期値)
    #hはfloat(微小時間)
    #Nはint(時間hNまで計算する)
    #Fはfloat(パラメータ)

    import numpy as np
    
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

#これは直接実行されたときにだけ実行され、他のスクリプトなどからimportなどにより実行される際には実行されないようにするための文
if __name__ == '__main__':

    X = Lorenz96_RK4([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8.008, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 0.01, 5000, 8.0)

    #Lorenz and Emanuelと同じ条件で計算する。


    #print(X)

    #実行するたびにXの値たちを書き換えたいので、前回以前の実行結果のファイルを削除しておく
    os.remove('../out.csv')
    #ファイルに計算結果を追記で記入するために開いておく
    with open('../out.csv','ab') as f:
        #ファイルに書き込み
        XX = np.array(X).reshape(5001,40)#こうやって整形しないとsavetxtで各要素ごとに改行される
        np.savetxt(f, XX, delimiter=",")

    import matplotlib.pyplot as plt

    for i in range(200,300):

        plt.plot(range(0,40), X[i])

    plt.show()

    #グラフを描画する。
    




