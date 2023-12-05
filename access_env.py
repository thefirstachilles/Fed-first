import math
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
from noma2oma import NOMA_method, OMA_method
def dbmtowatt(x):
    return (10**(x/10))/1000
def dbtowatt(x):
    return 10**(x/10)

class Sat_to_iot:
    def __init__(self, iot_num, rng, frame) -> None:
        self.h_list = []
        self.iot_num = iot_num
        self.rng = rng
        # 每一时隙最大的距离
        d_max = math.sqrt(7371**2 - 6371**2)*10**3/2
        # 每一时隙最小的距离
        d_min = 1000*10**3
        # 距离范围为
        for _ in range(frame):
            d = self.rng.random((self.iot_num))*(d_max-d_min)+d_min
            # 波长为光速/频率 波段范围在 30GHz左右
            wl = 3*10**8/(30*10**(9))
            # 加上自由衰落 ()^path_loss_exponent
            FPL = wl/(4*math.pi*d)**2.3
            # 加上雨衰2db, 发送增益10db 接收增益40db
            h = dbtowatt(10*np.log10(FPL)+11+43)
            self.h_list.append(h)
    #计算能耗
    def compute_energy(self, h):
        bandwidth = 200*10**6
        data_size = 30*8*10**3
        band_num = 2
        t_max =  5
        self.noma_output_energy = NOMA_method(h, bandwidth, data_size, band_num, t_max).output_energy
        self.oma_output_energy = OMA_method(h, bandwidth, data_size, t_max).output_energy
        
        print('noma_output_energy',self.noma_output_energy)
        print('oma_output_energy',self.oma_output_energy)
        # for h in self.h_list:
        #     noma = NOMA_method(h, bandwidth, data_size, band_num, t_max )
        #     print(noma.output_energy)
        #     oma = OMA_method(h, bandwidth, data_size,t_max)
        #     print(oma.output_energy)
        
        


if __name__=="__main__":
    # frame 为帧数
    frame = 30
    user_num = 10
    rng = np.random.default_rng(seed=50)
    sat = Sat_to_iot(user_num, rng, frame)
    for frame_num in range(frame):
        sat.compute_energy(sat.h_list[frame_num])
    
    
        