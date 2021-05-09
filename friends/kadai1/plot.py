#   author : Tan Moriteru
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/code/python/data-asmp/out.csv'
fig = plt.figure( figsize=(11, 5) )

F=8.0       # 方程式中の定数
J=40;   X = list(range(J+1))    # as [0, 1, ... , 40]
D = np.loadtxt(path, delimiter=',', dtype='float32')    # プロットするデータのinput

#N = len(D)             # データを全てプロットする用の部分
N=1+20;   D = D[:N]      # データの一部のみをプロットする用の部分

# X0 (=X40) を D に添加する
D = np.hstack((D[:, -1:], D))

# グラフのタイトルの設定
title = 'F = ' + str(F) +', Xj = F*1.001 if j = 20, else Xj = F'


# 論文中の fig1 タイプのグラフのプロット   
ax1 = fig.add_subplot(121)
ax1.set_title(title)  
ax1.set_xlabel('j = 1, 2, ... , 40')
ax1.set_ylabel('time step (day)')       # x/y 軸の名称設定
ax1.set_xlim(0, 40)
ax1.set_ylim(2.2, -0.2)                 # x/y 軸の範囲設定

#   プロット部
for i in range(9):
    ax1.axhline(y=i*0.25, color='gray', ls='--')
    ax1.plot(X, 5*(F-D[i]) +i*0.25, color='gray') 


# ホフメラー図のプロット
k=100;   h=0.001
L = [F*( 1+h*i ) for i in range(-k, k+1)]     # 等高線の高さの指定：F を中心に0.1%刻みで 90%～110%

ax2 = fig.add_subplot(122)
ax2.set_title(title)
ax2.set_xlabel('j = 1, 2, ... , 40')
ax2.set_ylabel('time step (day)')         # x/y 軸の名称設定

#   プロット部
Y = [i*0.25 for i in range(N)]
ax2.contourf(X, Y, D, levels=L, cmap='bwr')
ctr = ax2.contour(X, Y, D, levels=[L[i] for i in range(0, 2*k+1, 10)], cmap='bwr')
ax2.clabel(ctr, colors='black', fontsize='x-small')     # 高さの表示


plt.show()