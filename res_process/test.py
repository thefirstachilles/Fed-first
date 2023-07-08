import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import interp1d
with open("./res/Branch-Lstm-PPO_record.txt","r") as file:
    content = json.loads(file.read())
    process_content = []
    temp = 0
    for index, item in enumerate(content):
        temp+=item['reward']
        if index%10 ==0 and index != 0:
            process_content.append(temp)
            temp = 0
            
y=np.array(process_content)
x=np.array(list(range(0,len(y))))
plt.plot(x,y)
plt.show()