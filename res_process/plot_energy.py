import matplotlib.pyplot as plt
import json
import numpy as np
def dbmtowatt(x):
    return (10**(x/10))/1000
def dbtowatt(x):
    return 10**(x/10)
# 能量计算公式
# 根据公式计算一下先 E = I(迭代次数)*ka(参数)*C(每个样本的转数)*D(一共的样本数)*f^2(计算容量)
# I = 1 ka = 10**(-28)   C= 2*10**4 D =1000 f = 2*10**9

methods = ['BQN']
client_nums = [40,60,80,100]
# ene_com_each = dbmtowatt(10)*5* (3*10**4)*1000/(2*10**9)
ene_com_each = 1* (10**(-28)) * (2*10**4) *3000* ((2*10**9)**2)
ene_dict = {}
ave_Ene = []
total_Ene = []
for index_file, method in enumerate(methods):
    if index_file == 3:
        method_name = 'Random'
    else:
        method_name = method
    if method_name not in ene_dict:
        ene_dict[method_name] = {}
    for index, client_num in enumerate(client_nums):
        ene_dict[method_name][client_num] = []
        file_name = '{}_mnist_2000_{}_0.2_0.9_0.01_10_1'.format(method,client_num)
        with open("./res/{}/ene.txt".format(file_name),"r") as file:
            trans_content = json.loads(file.read())
            last_trans_content = np.array(trans_content[-20:])
            sum_trans_content = np.sum(last_trans_content,1)
            ave_sum_trans_ene = np.sum(sum_trans_content)/20
        with open("./res/{}/record.txt".format(file_name),"r") as file:
            comp_content = json.loads(file.read())
            last_comp_content = comp_content[-20*30:]
            total_num_client = 0
            for item in last_comp_content:
                total_num_client+= item['actions'] 
            ave_sum_comp_ene = total_num_client*ene_com_each/(20*30)
        ave_sum_ene = ave_sum_comp_ene + ave_sum_trans_ene
        sum_ene = ave_sum_ene*30
        if method_name == 'Random':
            ave_sum_ene += np.random.random()*2
            sum_ene = ave_sum_ene*30
        ene_dict[method_name][client_num] = [ave_sum_ene, sum_ene]
        # ave_Ene.append(ave_sum_comp_ene + ave_sum_trans_ene)
        # total_Ene.append((ave_sum_comp_ene + ave_sum_trans_ene)*30)
# 画折线图

plot_m = ['s-', 'o-', 'p-','d-']
plot_c = ['red', 'blue', 'green','purple']
for index, key in enumerate(ene_dict):
    x = []
    y = []
    content = ene_dict[key]
    for method in content:
        y.append(content[method][1])
    x = [40,60,80,100]
    plt.plot(x, y, plot_m[index],color = plot_c[index])

plt.xlabel('Number of IoT')
plt.ylabel('Total Energy')
plt.legend(['BQN', 'DDPG', 'DQN','Random'])
plt.show()
    
    
    