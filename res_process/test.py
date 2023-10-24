import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import interp1d
# file_list = ['BQN_mnist_2000_100_0.2_0.9_0.01_10_1','DDPG_mnist_2000_100_0.2_0.9_0.01_10_1']
# file_list = ['Branch-Lstm-PPO_mnist_2000_100_0.2_0.9_0.01_10_1']
file_list = ['BQN_mnist_2000_20_0.2_0.9_0.01_10_1']
# file_list = ['DDPG_mnist_2000_40_0.2_0.9_0.01_10_1','DDPG_mnist_2000_60_0.2_0.9_0.01_10_1','DDPG_mnist_2000_80_0.2_0.9_0.01_10_1','DDPG_mnist_2000_100_0.2_0.9_0.01_10_1']
# file_list = ['BQN_mnist_2000_100_0.2_0.9_0.01_10_1','DDPG_mnist_2000_100_0.2_0.9_0.01_10_1']

# file_list = ['DDPG_mnist_2000_40_0.2_0.9_0.01_10_1']

# file_list = ['DDPG_mnist_2000_40_0.2_0.9_0.01_10_1','BQN_mnist_2000_40_0.2_0.9_0.01_10_1']





for file_name in file_list:
    with open("./res/{}/record.txt".format(file_name),"r") as file:
        content = json.loads(file.read())
        process_content = []
        temp = 0
        index = 0
        for index, item in enumerate(content):
            # process_content.append(item['reward'])
            index+=1
            temp+=item['reward']
            if item['done'] == True:
                process_content.append(temp)
                temp = 0
                index = 0
                
        y=np.array(process_content[1:100])
        x=np.array(list(range(0,len(y))))
        plt.plot(x,y)
plt.legend(['1', '5', '10','15'])
plt.show()