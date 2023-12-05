import matplotlib.pyplot as plt
import json
import numpy as np
# from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
methods = ['BQN']

for index_file, file_name in enumerate(methods):
    
    with open("./res/method_{}/clientNum_30/iid_False/data_mnist_envFrame_30_rewardWeight_0.5_isblur_False_sampleNum_10_setSize_100_bqrLr_0.001/loss.txt".format(file_name),"r") as file:
         content = json.loads(file.read())
         candi_y1 = content[-20:-15]
        #  candi_y2 = content[-2]
         y1 = np.min(candi_y1,0)
         y2 = np.max(candi_y1,0)
         y3 = np.mean(candi_y1,0)
         x = np.array(list(range(0,len(y1))))
         
        #  X_Y_low_Spline = make_interp_spline(x, y1)
        #  X_low = np.linspace(x.min(), x.max(), 10)
        #  Y_low = X_Y_low_Spline(X_low)
         
        #  X_Y_high_Spline = make_interp_spline(x, y2)
        #  X_high = np.linspace(x.min(), x.max(), 10)
        #  Y_high = X_Y_high_Spline(X_high)
         
        #  X_Y_high_Spline = make_interp_spline(x, y3)
        #  X_mean = np.linspace(x.min(), x.max(), 10)
        #  Y_mean = X_Y_high_Spline(X_mean)
         
         
        #  plt.fill_between(X_low, Y_low,Y_high, alpha=0.5)
        #  plt.plot(X_mean,Y_mean)
         plt.fill_between(x, y1,y2, alpha=0.5)
         plt.plot(x,y3)
plt.legend(methods)
plt.show()
    