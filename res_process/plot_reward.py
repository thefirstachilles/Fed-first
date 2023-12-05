import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import interp1d
# file_list = ['BQN_mnist_2000_100_0.2_0.9_0.01_10_1','DDPG_mnist_2000_100_0.2_0.9_0.01_10_1']
# file_list = ['Branch-Lstm-PPO_mnist_2000_100_0.2_0.9_0.01_10_1']
# file_list = ['BQN_mnist_2000_20_0.2_0.9_0.01_10_1']
# file_list = ['DDPG_mnist_2000_40_0.2_0.9_0.01_10_1','DDPG_mnist_2000_60_0.2_0.9_0.01_10_1','DDPG_mnist_2000_80_0.2_0.9_0.01_10_1','DDPG_mnist_2000_100_0.2_0.9_0.01_10_1']
# file_list = ['BQN_mnist_2000_100_0.2_0.9_0.01_10_1','DDPG_mnist_2000_100_0.2_0.9_0.01_10_1']

# file_list = ['DDPG_mnist_2000_40_0.2_0.9_0.01_10_1']

# file_list = ['DDPG_mnist_2000_40_0.2_0.9_0.01_10_1','BQN_mnist_2000_40_0.2_0.9_0.01_10_1']

# file_list = ['BQN','DDPG', 'Random']
# file_list = ['BQN','DDPG']
file_list = ['BQN']

# config_list = ['bqrLr_0.001', 'lr_a_0.001_lr_c_0.002', '_']
# config_list = ['bqrLr_0.01', 'lr_a_0.0001_lr_c_0.002']
config_list = ['selectNum_variable_bqrLr_0.001']

# /Users/zhouzhou/Fed-first/res/method_DDPG/clientNum_20/iid_False/data_mnist_envFrame_5_rewardWeight_0.5_isblur_False_sampleNum_10_setSize_100_punish_30_lr_a_0.001_lr_c_0.002/

# with open("./res/method_{}/clientNum_30/iid_False/data_mnist_envFrame_30_rewardWeight_0.5_isblur_False_sampleNum_10_setSize_100_bqrLr_0.001/record.txt".format(file_name),"r") as file:

# with open("./res/method_{}/clientNum_20/iid_False/data_mnist_envFrame_30_rewardWeight_0.5_isblur_False_sampleNum_10_setSize_100_punish_30_lr_a_0.001_lr_c_0.002/record.txt".format(file_name),"r") as file:

for index_file, file_name in enumerate(file_list):
    # for index, bqn_lr in enumerate(config_list):
    # with open("./res/method_{}/clientNum_30/iid_False/data_mnist_envFrame_30_rewardWeight_0.5_isblur_False_sampleNum_10_setSize_100_bqrLr_0.001/record.txt".format(file_name),"r") as file:
        with open("./res/method_{}/clientNum_40/iid_False/data_mnist_envFrame_30_rewardWeight_0.5_isblur_False_sampleNum_10_setSize_100_punish_60_rewardMethod_variable_{}/record.txt".format(file_name, config_list[index_file]),  "r") as file:
            content = json.loads(file.read())
            process_content = []
            temp = 0
            inner_index = 0
            for index, item in enumerate(content):
                # process_content.append(item['reward'])
                temp+=item['reward']
                inner_index+=1
                if item['done']==True and inner_index== 30:
                    process_content.append(temp)
                    temp = 0
                    inner_index = 0
                elif item['done']==True:
                    temp = 0
                    inner_index = 0
                        
                    
            y=np.array(process_content[1:])
            x=np.array(list(range(0,len(y))))
            plt.plot(x,y)
# plt.ylim(-350)
plt.title('80 IoT')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(config_list)
plt.show()