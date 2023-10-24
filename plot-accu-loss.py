import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import interp1d


with open("./test_reward_accu.txt" ,"r") as file:
    content = json.loads(file.read())
    loss = []
    delta_loss = []
    accu = []
    delta_accu = []
    reward = []
    for index, item in enumerate(content[:50]):
        loss.append(item['test_loss'])
        delta_loss.append(item['delta_loss'])
        accu.append(item['global_accu'])
        delta_accu.append(item['delta_accu'])
        if item['last_loss']>0.5:
            reward.append(item['delta_loss'] - item['last_loss']/1.5)
        else:
            reward.append(item['delta_loss'] - 0.1)
        # reward.append((loss_item+0)*(index+1)*10/7)
        # reward.append((1/loss_item))
        # A = 0.786254763206445
        # B = 0.641618966371729
        # C = 1.5164626038415
        # D = 0.100341908272168
        # # reward.append((loss_item-(((A - D) / (1 + (index/C)**B)) + D))+10)
        # if index<3:
        #     reward.append(0)
        # else:
        #     reward.append(((1/(index+0.85)**2)-item['delta_loss']))
       
        
        
        # if index< 10:
        #     a = 2.25209022319399
        #     b = 0.0703665527500099
        #     c = -0.0725867426513563
        #     d = 0.00518770741346589
        #     reward.append(10*loss_item/(a+b*index+c*index**2+d*index**3))
        # else:
        #     a = 1.03985939537506
        #     b = -0.0422685414545357
        #     c = 0.000976547034379296
        #     d = -7.95166620691345*10**(-6)
        #     reward.append(10*loss_item/(a+b*index+c*index**2+d*index**3))
        # reward.append(loss_item/((2.3-0.35)/(1+(index/5.265)**1.924)+0.3433))
        
loss=np.array(loss)
delta_loss=np.array(delta_loss)
accu=np.array(accu)
delta_accu=np.array(delta_accu)
reward = np.array(reward)
x=np.array(list(range(0,len(loss))))
plt.plot(x,loss)
plt.plot(x,delta_loss)
plt.plot(x,accu)
plt.plot(x,delta_accu)
plt.plot(x,reward)
plt.legend(['loss', 'delta_loss', 'accu','delta_accu','reward'])
plt.show()
    