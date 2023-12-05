import math
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
# db与p的转换
def dbmtowatt(x):
    return (10**(x/10))/1000
def dbtowatt(x):
    return 10**(x/10)
deg2rad=np.deg2rad
rad2deg=np.rad2deg
arctan=np.arctan
arcsin=np.arcsin
sin=np.sin
cos=np.cos
pi = np.pi
log = np.log2
C = 3*10**8
f = 30*10**9
d_s = 200*10**(-3)
ka = 1.380649*10**(-23)
B = 10*10**6
class Sat_to_iot:
    def __init__(self, iot_num, rng) -> None:
        self.iot_num = iot_num
        # """卫星轨道，分为N个时隙，每一个时隙中，卫星的位置发生变化"""
        # 参考：Doppler Characterization for LEO Satellites
        # 东西长1000km，即是说经度差约9度，维度就放赤道上怎么了？那就也是9°
        # 经度是40°到49°，维度是南纬4.5°到北纬4.5° ，那南北纬可以重合算嘛
        self.sate_h=1000*10**3 # 高度1000km 
        self.r_e = 6371*10**3 #地球半径6371km
        theta_yuanxin=self.sate_h*180/pi/self.r_e
        # 假设iot经纬度
        lo_st=40
        la_st=0
        lo_rand=rng.random(self.iot_num)*theta_yuanxin+lo_st
        la_rand=rng.random(self.iot_num)*theta_yuanxin/2+la_st
        self.node_loca=np.stack((lo_rand,la_rand),axis=1)
        # 卫星初始经纬度
        sate_lo=25.01   
        sate_la=0
        self.sate_loca=np.array([sate_lo,sate_la])  #卫星位置

        g=9.8 #加速度 m/s
        #卫星角速度 
        self.w_v=rad2deg(((g*self.r_e**2)/((self.r_e+self.sate_h)**3))**(0.5)) #-7.292*10**(-5)
        # 轨道倾斜角
        self.inclin = deg2rad(0) # 单位度转换为rad
        self.E_client = 0.00001*np.ones(iot_num)
        self.total_E = 0
    def out_geometry(self):
        # 逐秒计算卫星经度差和维度差，并且存储
        # 以10s 为一个slot
        t =25
        # self.d_s_arr = []
        if self.sate_loca[0] > 80:
            # 重新初始化卫星初始经纬度
            sate_lo=25.01   
            sate_la=0
            self.sate_loca=np.array([sate_lo,sate_la]) 
        # 更新卫星经度与纬度
        delta_lo_la = np.array([t*self.w_v*cos(self.inclin), t*self.w_v*sin(self.inclin)])
        self.sate_loca = self.sate_loca + delta_lo_la
            #卫星和地面节点的经度差和维度差
        self.comp_sate_node=self.node_loca-self.sate_loca
        # 单位转化
        temp_comp_sate_node=deg2rad(self.comp_sate_node)
        temp_sate_lo=deg2rad(self.sate_loca[0])
        temp_sate_la=deg2rad(self.sate_loca[1])
        up=cos(np.squeeze(temp_comp_sate_node[:,0]))*cos(np.squeeze(temp_comp_sate_node[:,1]))*cos(temp_sate_la)+sin(np.squeeze(temp_comp_sate_node[:,1]))*sin(temp_sate_la)-self.r_e/(self.r_e+self.sate_h)
        down=(1-(sin(np.squeeze(temp_comp_sate_node[:,1]))*sin(temp_sate_la)+cos(np.squeeze(temp_comp_sate_node[:,0]))*cos(np.squeeze(temp_comp_sate_node[:,1]))*cos(temp_sate_la))**2)**0.5
        ## 仰视角
        self.alpha_sate=rad2deg(arctan(up/down))
        # self.sat_alpha_arr.append(alpha_sate)
        #卫星和地面点的距离 三角形余弦公式 a**2=b**2+c**2-2*b*c*cosA
        self.d_n=((self.r_e+self.sate_h)**2+(self.r_e)**2-2*(self.r_e+self.sate_h)*self.r_e*np.cos(np.squeeze(temp_comp_sate_node[:,0])))**0.5
        # self.d_n_arr.append(d_n)
        # 卫星与波束圆心的距离
        # d_s = ((1-((sin(deg2rad(alpha_sate+90))*d_n)/(self.r_e+self.sate_h))**2)**0.5)*d_n
        # self.d_s_arr.append(d_s)

        return
    def out_h(self):
        # 本轮要计算的
        ## 根据参数计算信道信息，并且存储信息
        
        fin_k = (pi*d_s*f/C)*sin(deg2rad(self.alpha_sate))
        J_1 = special.jn(1, fin_k)
        J_3 = special.jn(3, fin_k)
        omege_k = (J_1/(2*fin_k))+36*(J_3/(fin_k**3))
        G_k_T = dbtowatt(34)
        r_k = (10**(np.exp((np.random.normal(loc=dbtowatt(-2.6), scale= dbtowatt(1.63), size=(self.iot_num, 2)).view(np.complex128)))/20)).flatten()
        FPL = (C/(4*pi*self.d_n*f))**2
        # g_k = FPL*(G_k_T/(ka*B))*(1/r_k)
        g_k = FPL*G_k_T/(ka*B)
        h_k_los = g_k**(0.5)
        self.h_k_los_value = np.absolute(h_k_los)

    def get_power(self, clients):
        ## 计算信道容量，并且存储
        ## 传输的信息大小有30k byte
        M = 30*8*10**3
        noise_power = dbmtowatt(30)
        p_max = 1
        C_k = B*log(1 + p_max*(self.h_k_los_value[clients]**2)/(noise_power))
        self.E_client = M/C_k*p_max
        self.total_E = np.sum(self.E_client)
        # capacity = 1+np.log(,2)

        return

if __name__=="__main__":
    rng = np.random.default_rng(seed=50)
    sattoiot = Sat_to_iot(100, rng)
    energy_list = []
    while sattoiot.sate_loca[0]>25 and sattoiot.sate_loca[0]<80:
        sattoiot.out_geometry()
        sattoiot.out_h()
        sattoiot.get_power([1,2,3])
        energy_list.append(sattoiot.total_E)
    plt.plot(energy_list)
    plt.show()






    
        