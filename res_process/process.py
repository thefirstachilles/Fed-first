import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
# 取文件内容
file_list = ['mnist_random_1000_15_1_0.9_0.001_10_1_False_acu','mnist_async_2000_12_0.2_0.9_0.001_10_1_False_acu','mnist_isl_2000_12_0.2_0.9_0.001_10_1_False_acu']
file_list_2 = ['mnist_random_2000_12_0.2_0.9_0.001_10_1_True_acu','mnist_async_2000_12_0.2_0.9_0.001_10_1_True_acu', 'mnist_isl_2000_12_0.2_0.9_0.001_10_1_True_acu']
file_list_3 = ['cifar_random_2000_12_0.4_0.9_0.001_10_1_True_acu','cifar_async_2000_12_0.2_0.9_0.001_10_1_True_acu', 'cifar_isl_2000_12_0.2_0.9_0.001_10_1_True_acu','cifar_myapproach_2000_12_0.2_0.9_0.001_10_1_True_acu']
file_list_4 = ['cifar_random_2000_12_0.2_0.9_0.001_10_1_False_acu','cifar_async_2000_12_0.2_0.9_0.001_10_1_False_acu', 'cifar_isl_2000_12_0.2_0.9_0.001_10_1_False_acu','cifar_myapproach_2000_12_0.4_0.9_0.001_10_1_False_acu']
file_list_5 = ['cifar_random_1000_50_0.3_0.9_0.001_10_1_False_acu','cifar_random_1000_50_0.3_0.9_0.001_10_5_False_acu','cifar_random_1000_50_0.3_0.9_0.001_10_10_False_acu']

for file_path in file_list_5:
    with open("./res/{}.txt".format(file_path),"r") as file:
        content = json.loads(file.read())
        file.close()
        #第1步：定义x和y坐标轴上的点  x坐标轴上点的数值
        y=np.array(content['memo_dict'])
        if file_path == 'mnist_acu':
            y=y[0::5]
        else:
            y = y[0:200]
        #y坐标轴上点的数值
        x=np.array(list(range(0,len(y))))
        # y_smoothed = gaussian_filter1d(y, sigma=5)
        # model=interp1d(x, y)

        # xs=np.linspace(0,999,500)
        
        # ys=model(xs)
        #第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值
        plt.plot(x,y)
plt.legend(['1', '5', '10'])
#添加文本 #x轴文本
plt.xlabel('x坐标轴')
#y轴文本
plt.ylabel('y坐标轴')
#标题
plt.title('标题')
#添加注释 参数名xy：箭头注释中箭头所在位置，参数名xytext：注释文本所在位置，
#arrowprops在xy和xytext之间绘制箭头, shrink表示注释点与注释文本之间的图标距离

# plt.annotate('我是注释', xy=(2,5), xytext=(2, 10),
#             arrowprops=dict(facecolor='black', shrink=0.01),
#             )

#第3步：显示图形
plt.show()

# x=np.array([1,2,3,4,5,6,7])
# y=np.array([100,50,25,12.5,6.25,3.125,1.5625])

# model=make_interp_spline(x, y)

# xs=np.linspace(1,7,500)
# ys=model(xs)