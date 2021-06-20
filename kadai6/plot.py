import numpy as np
import matplotlib.pyplot as plt

path_kf =  '../kf_rmse.csv'
path_oi = '../oi_rmse.csv'

# プロットするデータのinput
Dkf = np.loadtxt(path_kf, delimiter=',', dtype='float32')
# プロットするデータのinput
Doi = np.loadtxt(path_oi, delimiter=',', dtype='float32')

plt.plot(Dkf,label="KF(inflation=1.1)")
plt.plot(Doi,label="3Dvar")
plt.title("観測点を減らしていくときのRMSEのふるまい",fontname="MS Gothic")
plt.xlabel("観測点数",fontname="MS Gothic")
plt.ylabel("RMSE of analysis",fontname="MS Gothic",fontsize=15)
plt.legend()
plt.show()