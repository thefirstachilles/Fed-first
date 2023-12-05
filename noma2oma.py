import numpy as np
import matplotlib.pyplot as plt
## 编写两个接入方法，一个是oma 一个是优化了的noma方法， 进行能耗对比

##计算中的辅助函数
def dbmtowatt(x):
    return (10**(x/10))/1000
def dbtowatt(x):
    return 10**(x/10)

class NOMA_method:
    def __init__(self, h, bandwidth, data_size, band_num, t_max):
        self.h = h
        # 分固定的频谱资源块
        self.band_num = band_num
        self.each_bandwidth = bandwidth/self.band_num
        self.data_size = data_size
        # 噪声频谱
        self.noise_density=dbmtowatt(-174)
        # 最大时延
        self.t_max = t_max
        # 最大功率
        self.p_max = dbmtowatt(23)
        self.clusters = [[] for _ in range(band_num)]
        self.decide_cluster()
        # 参数步长
        self.step_param1 = None
        self.step_param2 = None
        # 变量步长
        self.step_var_z = None
        # 最大迭代次数
        self.ite_num = 1000
        # 初始输出能耗
        self.output_energy = 0
        for cluster in self.clusters:
            # user_num = len(cluster)
            if len(cluster)>0:
                self.get_op_p(cluster)
                self.output_energy += self.temp_energy
        self.output_energy
    def decide_cluster(self):
        h_max = np.max(self.h)
        h_min = np.min(self.h)
        h_indices = np.argsort(self.h)
        
        for index, h_ele in enumerate(self.h):
            cluster_index = h_indices[index] % self.band_num
            self.clusters[cluster_index].append(h_ele)
            
    def get_op_p(self, cluster):
        # 簇内用户数
        self.usr_num = len(cluster)
        # 参数初始设置
        self.param1 = np.ones(self.usr_num)*10**0
        self.param2 = np.ones(self.usr_num)*10**0
        # self.param3 = np.ones(self.usr_num)*10**0
        # 变量初始设置
        
        #在时延范围内的信噪比
        z_temp=2**(self.data_size/self.t_max/self.each_bandwidth)-1
        
        # 初始值p 设置为最大p的一半
        # self.p = np.ones(self.usr_num)*self.p_max
        self.p = np.ones(self.usr_num)*self.p_max/100
        # 根据初始p计算初始z 和 c, 按照一定的解码顺序重新排列h，这里默认为按照为从大到小排列，先解码最大的，最大的会有更小的作为干扰
        self.SIC_h = np.sort(cluster)[::-1]
        self.z = np.zeros(self.usr_num)
        self.compute_z()
        # for index, h_ele in  enumerate(self.SIC_h):
        #     self.z[index] =h_ele*self.p[index]/(self.noise_density*self.each_bandwidth+ np.sum(self.SIC_h[index+1:]*self.p[index+1:]))
        self.z_piao = np.log(self.z)
        # self.c_piao = np.log((self.each_bandwidth*np.log2(1+self.SIC_h*self.p/(self.noise_density*self.each_bandwidth))))
        # 参数步长初始设置
        self.step_param1 = np.random.rand(self.usr_num)*10**-1
        self.step_param2=np.random.rand(self.usr_num)*10**-1
        # self.step_param3=np.random.rand(self.usr_num)*10**0
        # 变量步长初始设置
        # self.step_c=np.ones(self.usr_num)*10**-3
        self.step_z=np.ones(self.usr_num)*10**-3
        self.ene_list = []
        
        init_energy = self.data_size
        for ite in range(self.ite_num):
            # 计算 multi
            self.multi = np.zeros(self.usr_num)
            self.compute_multi()
            # 计算 add
            self.add = np.zeros(self.usr_num)
            self.compute_add()
            # 计算 sum_langmuda
            self.sum_param1 = np.zeros(self.usr_num)
            self.compute_sum_param1()
            # 原始目标函数（没有ln）
            f_function = sum(self.data_size*self.noise_density*np.exp(self.z_piao)*self.multi/(self.SIC_h*np.log2(1+np.exp(self.z_piao))))
            delta_z_piao = self.noise_density/f_function*(self.data_size/self.SIC_h*((np.exp(self.z_piao)/(np.log2(1+np.exp(self.z_piao))))-(((np.exp(self.z_piao)**2)/(np.log(2)*(1+np.exp(self.z_piao))*(np.log2(1+np.exp(self.z_piao)))**2))))*self.multi+self.add)+self.param1+np.exp(self.z_piao)/(1+np.exp(self.z_piao))*self.sum_param1-self.param2*np.exp(self.z_piao)/self.each_bandwidth*(1+np.exp(self.z_piao)*np.log2(1+np.exp(self.z_piao))*np.log(2))
            # delta_c_piao = -self.data_size*self.each_bandwidth*self.noise_density/f_function*(np.exp(self.z_piao-self.c_piao)*self.multi)-self.param3- self.param2
            
            self.step_z = np.ones(self.usr_num)*1/(np.max(delta_z_piao)*1000)
            # 使用delta 更新变量
            self.z_piao = self.z_piao - self.step_z * delta_z_piao
            # self.c_piao = self.c_piao - self.step_c * delta_c_piao
            
            self.add_logex = np.zeros(self.usr_num)
            self.compute_add_logex()
            # 得到拉格朗日系数的delta
            delta_param1 = self.z_piao+self.add_logex-np.log(self.p_max*self.SIC_h/(self.each_bandwidth* self.noise_density))
            delta_param2 = -np.log(self.t_max/self.data_size)-np.log(self.each_bandwidth*np.log2(1+np.exp(self.z_piao)))
            # 更新拉格朗日系数
            self.param1 = self.param1 + self.step_param1*delta_param1
            self.param2 = self.param2 + self.step_param2*delta_param2
            for index, ele in enumerate(self.param1):
                if ele<=0:
                    self.param1[index] = 0
            for index, ele in enumerate(self.param2):
                if ele<=0:
                    self.param2[index] = 0
            # 收集这一轮的解
            self.temp_z = np.exp(self.z_piao)
            self.multi_temp_z = np.zeros(self.usr_num)   
            self.compute_multi_temp_z()
            self.temp_p = self.noise_density*self.each_bandwidth*self.temp_z*self.multi_temp_z/self.SIC_h
            self.temp_t = self.data_size/(self.each_bandwidth*np.log2(1+self.SIC_h*self.temp_p/(self.noise_density*self.each_bandwidth)))
            self.temp_energy = f_function
            self.ene_list.append(f_function)
            if ite%500 == 0:
                continue
        # plt.plot( list(range(len(self.ene_list))), self.ene_list)
        # plt.show()
    def compute_z(self):
        for n_index , h in enumerate(self.SIC_h):
            multi_item = 0
            for j_index, h in enumerate(self.SIC_h[n_index+1:]):
                multi_item = multi_item + self.p[j_index]*self.SIC_h[j_index]
            self.z[n_index] = self.p[n_index] * self.SIC_h[n_index]/(self.noise_density*self.each_bandwidth+multi_item)
    def compute_multi(self):
        for n_index, h in enumerate(self.SIC_h):
            n_item = 1
            for j_index, h in enumerate(self.SIC_h[n_index+1:]):
                n_item = n_item*(1 + np.exp(self.z_piao[j_index]))
            self.multi[n_index] = n_item
    def compute_add(self):
        for n_index, h_n in enumerate(self.SIC_h):
            # 累加项 each_item 里有两项相乘，分别计为前项和后项
            total_item = 0
            # 计算前一项
            for i_index, h_i in enumerate(self.SIC_h[:n_index]):
                front_item = self.data_size/self.SIC_h[n_index]*np.exp(self.z_piao[i_index]+self.z_piao[n_index])/(np.log2(1+np.exp(self.z_piao[i_index])))
                back_item = 1
                for j_index, h_i in enumerate(self.SIC_h[i_index+1:]):
                    if j_index == n_index:
                        continue
                    back_item = back_item*(1+np.exp(self.z_piao[j_index]))
                each_item = front_item*back_item
                total_item = total_item + each_item
            self.add[n_index] = total_item
    def compute_sum_param1(self):
        for n_index, h_n in enumerate(self.SIC_h):
            n_item = 0
            for i_index,h_i in enumerate(self.SIC_h[:n_index]):
                n_item = n_item + self.param1[i_index]
            self.sum_param1[n_index] = n_item
    def compute_add_logex(self):
        for n_index, h_n in enumerate(self.SIC_h):
            n_item = 0
            for j_index, h_j in enumerate(self.SIC_h[n_index+1:]):
                n_item = n_item + np.log10(1 + np.exp(self.z_piao[j_index]))
            self.add_logex[n_index] = n_item
    def compute_multi_temp_z(self):
        for n_index, h_n in enumerate(self.SIC_h):
            n_item = 1
            for j_index, h_j in enumerate(self.SIC_h[n_index+1:]):
                n_item = n_item*(1+self.temp_z[j_index])
            self.multi_temp_z[n_index] = n_item
class OMA_method:
    def __init__(self, h, bandwidth, data_size,t_max):
        self.h = h
        # 每个用户平均分配频谱资源块
        self.band_num = len(h)
        self.each_bandwidth = bandwidth/self.band_num
        # 最大功耗
        self.p = dbmtowatt(23)
        # 最大时间
        self.t_max = t_max
        # 噪声频谱
        self.noise_density=dbmtowatt(-174)
        # 数据长度 100kb
        self.data_size = data_size
        # self.c = self.data_size/self.t_max
        self.c = self.each_bandwidth*np.log2(1+self.h*self.p/(self.noise_density*self.each_bandwidth))
        self.compute_output_energy()
    def compute_output_energy(self):
        self.T = self.data_size/self.c
        # self.p = ((2**(self.c/self.each_bandwidth)-1)*(self.noise_density*self.each_bandwidth))/self.h
        self.output_energy = np.sum(self.p * self.T)


        

if __name__=="__main__":
    np.random.seed(5)
    h1 = 10**(-14)*np.random.random(10)
    bandwidth = 200*10**6 
    data_size = 30*8*10**3
    t_max = 10
    oma = OMA_method(h1, bandwidth, data_size,t_max)
    print('oma',oma.output_energy)
    band_num = 1
    noma = NOMA_method(h1, bandwidth, data_size, band_num, t_max)
    print('noma',noma.output_energy)