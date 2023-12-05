import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import os

#改成卫星链路
##计算中的辅助函数
def dbmtowatt(x):
    return (10**(x/10))/1000
def dbtowatt(x):
    return 10**(x/10)

def compute_x(SIC,array,ChannelGain):
    x=[]
    for idx,data in enumerate(SIC):
#         print("idx",idx) #顺序
#         print("data",data) #节点，后面就是噪声
        temp=0
        for item in SIC[idx+1:]:
            temp=temp+array[item]*ChannelGain[item]
        x.append(temp)
    x=np.array(x)         
    return x

def compute_ex(SIC,array):
    x=[]
    for idx,data in enumerate(SIC):
        temp=1
        for item in SIC[idx+1:]:
            temp=temp*(1+np.exp(array[item]))
        x.append(temp)
    x=np.array(x)
        
    return x
def compute_add(SIC,array1,array2,array3,array4):
    x=[]
    for idx,data in enumerate(SIC):
        temp1=0
        for item in SIC[:idx]:
            temp1=temp1+array1[item]/array2[item]*np.exp(array3[item]+array3[idx]-array4[item])
            temp2=1
            for item2 in SIC[item+1:]:
                if not item2==idx:
#                     print("array3[item2]",array3[item2])
                    temp2=temp2*(1+np.exp(array3[item2]))
#                     print("idx",idx,"item",item,"item2",item2)
                tem1=temp1*temp2
        x.append(temp1)
    x=np.array(x)


    return x


def compute_langmuda(SIC,array):
    x=[]
    for idx,data in enumerate(SIC):
        temp=0
        for item in SIC[:idx]:
            temp=temp+array[item]
        x.append(temp)
    x=np.array(x)
    return x

def compute_c(SIC,array):
    x=[]
    for idx,data in enumerate(SIC):
        temp=0
        for item in SIC[:]:
            temp=temp+np.exp(array[item])
        x.append(temp)
    x=np.array(x)
    return x
def compute_logex(SIC,array):
    x=[]
    for idx,data in enumerate(SIC):
        temp=0
        for item in SIC[idx+1:]:
            temp=temp+np.log(1+np.exp(array[item]))
        x.append(temp)
    x=np.array(x)
    return x

def compute_ec(SIC,array):
    x=[]
    for idx,data in enumerate(SIC):
        temp=0
        for item in SIC[idx+1:]:
            temp=temp+np.exp(array[item])
        x.append(temp)
    x=np.array(x)
    return x
def compute_multix(SIC,array):
    multix=[]
    for idx,data in enumerate(SIC):
        temp=1
        for item in SIC[idx+1:]:
            temp=temp*(1+array[item])
        multix.append(temp)
    multix=np.array(multix)
        
    return multix

class op_p:
    def __init__(self,num_node,D_n,SIC):
        #节点数目
        self.num_node=num_node
        # 设置的SIC顺序
        self.SIC=SIC
        node=np.arange(self.num_node)
        #发射天线增益和接收天线增益
        G_T=45
        G_R=50
        f_c=20*10**9
        #增益，大尺度衰落
        self.ChannelGain=np.array([dbtowatt(G_T+G_R-(28+20*np.log10((d/10**3)*(f_c/10**3)))) for d in D_n])
        #带宽
        self.BandWidth=10**6 
        #噪声
        self.Noise=dbmtowatt(-174)*self.BandWidth
        # 最大传输功率
        self.max_allowable_transmit_power=dbmtowatt(23)
        #x的取值范围（好像是最大的可能的信噪比，当功率最大，信道状态最好，也没有其他弱信号干扰时，信噪比最大）
        max_x=np.log(self.max_allowable_transmit_power*self.ChannelGain[-1]/self.Noise)
        min_x=0
        # 传输数据大小
        max_B=10**5
        self.B=np.ones(self.num_node)*max_B*np.log(2) #单位 Mb
        #单位 second,限制的时延
        self.T=10 
        
        ##如果用oma的方法，那就是继续分带宽，然后每个带宽里最大速率传输
        self.oma_cap = self.BandWidth/3*np.log(1+((self.max_allowable_transmit_power*self.ChannelGain)/(dbmtowatt(-174)*self.BandWidth/3)))
        self.oma_ene = self.max_allowable_transmit_power *max_B/(self.oma_cap)
        self.sum_oma_ene = np.sum(self.oma_ene)
        
        
        #在时延范围内的信噪比
        x_temp=np.exp(self.B/self.T/self.BandWidth)-1
        x=np.ones(self.num_node)*(x_temp.max())+1
        self.max_DataRate=self.BandWidth*np.log(1+x)
        # 最大传输速率初始化
        R=np.random.rand(self.num_node)*self.max_DataRate/2
         #参数步长等初始设置
        self.langmuda=np.random.rand(self.num_node)*10**0
        self.fin=np.random.rand(self.num_node)*10**0
        self.gamma=np.random.rand(self.num_node)*10**0
        
        # 变量步长初始设置
        self.x_size=np.ones(self.num_node)*10**-3
        self.R_size=np.ones(self.num_node)*10**-2
        self.step_size_k=np.ones(3)*10**0

        # 替换变量
        self.x=np.log(x)
        self.R=np.log(R)
        
        
        # 迭代次数
        self.frame_num=10**4
        #记录结果列表
        self.p_list=[]
        self.x_list=[]
        self.R_list=[]
        self.T_list=[]
        self.energy_list=[]
        self.crite_list=[]
        
        
        self.get_op_p()
        
        
        
        
    def compute_f(self):
        
        multi=compute_ex(self.SIC,self.x)
        add=compute_add(self.SIC,self.B,self.ChannelGain,self.x,self.R)
        sum_langmuda=compute_langmuda(self.SIC,self.langmuda)
        add_logex=compute_logex(self.SIC,self.x)     
        self.f_function=sum(self.B*self.Noise*np.ones(self.num_node)*(np.exp(self.x-self.R))/self.ChannelGain*multi)
        self.delta_x=self.Noise/self.f_function*(self.B/self.ChannelGain*np.exp(self.x-self.R)*multi+add+self.langmuda+np.exp(self.x)/(1+np.exp(self.x))*sum_langmuda+self.fin*np.exp(self.x)/(self.BandWidth*(np.log(1+np.exp(self.x)))*(1+np.exp(self.x))))
        self.delta_R=-self.B*self.Noise/(self.f_function*self.ChannelGain)*np.exp(self.x-self.R)*multi-self.gamma*np.exp(-self.R)/(np.exp(-self.R))+self.fin
#         print("f_function",self.f_function)
#         print("delta_x",self.delta_x)
#         print("delta_R",self.delta_R)

    def update_var(self):
       
        add_logex=compute_logex(self.SIC,self.x)
        delta_langmuda=self.x+add_logex-np.log(self.max_allowable_transmit_power*self.ChannelGain/self.Noise)
        delta_fin=self.R-np.log(self.BandWidth*np.log(1+np.exp(self.x)))
        delta_gamma=np.log(np.exp(-self.R))-np.log(self.T/self.B)
        self.langmuda=self.langmuda+self.step_size_k[0]*delta_langmuda
        self.fin=self.fin+self.step_size_k[1]*delta_fin
        self.gamma=self.gamma+self.step_size_k[2]*delta_gamma
#         print("delta_langmuda",delta_langmuda)
#         print("delta_fin",delta_fin)
#         print("delta_gamma",delta_gamma)
        for idx ,data in enumerate(self.langmuda):
            if data < 0:
                self.langmuda[idx]=0

        for idx ,data in enumerate(self.fin):
            if data < 0:
                self.fin[idx]=0

        for idx ,data in enumerate(self.gamma):
            if data < 0:
                self.gamma[idx]=0
#         print("langmuda",self.langmuda)
#         print("fin",self.fin)
#         print("gamma",self.gamma)
        

            
        
    def update_x(self):
        self.x=self.x-self.x_size*self.delta_x
        self.R=self.R-self.R_size*self.delta_R
#         print("x",self.x)
#         print("R",self.R)
        return_x=np.exp(self.x)
        return_R=np.exp(self.R)
        max_rate=self.BandWidth*np.log(1+return_x)
        for idx ,data in enumerate(max_rate):
            if data <return_R[idx]:
                return_R[idx]=data
        
        self.R=np.log(return_R)
        
        
    
    def validate_x(self):
        a=self.delta_x
        b=self.delta_R
        
        c=np.stack((a,b), axis=1)
        c=np.split(c, self.num_node)
        c=np.array([np.squeeze(data) for data in c])
        crite=np.array([np.linalg.norm(data, axis=0) for data in c])
#         print("crite",crite,crite.mean())
        self.crite_list.append(crite.mean())
        
            
        
    def return_energy(self):
        out_x=np.exp(self.x)
        out_R=np.exp(self.R)
        self.max_rate=self.BandWidth*np.log(1+out_x)
        multix=compute_multix(self.SIC,out_x)
#         print("x",out_x)
        outcome_p=self.Noise*out_x*multix/self.ChannelGain        
        outcome_R=out_R
        outcome_T=self.B/out_R
        out_energy=sum(self.B*self.Noise*multix/(self.ChannelGain*out_R))
        
        self.x_list.append(out_x)
        self.p_list.append(outcome_p)
        self.R_list.append(outcome_R)
        self.T_list.append(outcome_T)
        self.energy_list.append(out_energy)
#         print("outcome_p",outcome_p)
#         print("outcome_R",outcome_R)
#         print("outcome_T",outcome_T)
        
#         print("out_energy",out_energy)
        
    def plot_image(self):
        # plt.plot(self.crite_list)
        # plt.show()
        # plt.plot(self.energy_list)
        # plt.show()
        self.p_list=np.array(self.p_list)
        plt.plot(self.p_list[:,0])
        plt.plot(self.p_list[:,1])
        plt.plot(self.p_list[:,2])
        plt.plot(np.ones(self.frame_num)*self.max_allowable_transmit_power)
        plt.show()
        # self.T_list=np.array(self.T_list)
        # plt.plot(self.T_list[:,0])
        # plt.plot(self.T_list[:,1])
        # plt.plot(self.T_list[:,2])
        # plt.plot(np.ones(self.frame_num)*self.T)
        # plt.show()
    
    def find_the_small(self):
        the_idx=None
        self.p_list=np.array(self.p_list)
        for i in range(self.frame_num):
            if self.p_list[i,0]<self.max_allowable_transmit_power and self.p_list[i,1]<self.max_allowable_transmit_power and self.p_list[i,2]<self.max_allowable_transmit_power :
                the_idx1=i
                break
        self.T_list=np.array(self.T_list)
        for j in range(self.frame_num - 1, -1, -1):
            if self.T_list[j,0]<self.T and self.T_list[j,1]<self.T and self.T_list[j,2]<self.T :
                the_idx2=j
                break
#         print(the_idx)
        the_idx=min(the_idx1,the_idx2)
        the_x=self.x_list[the_idx]
        the_R=self.R_list[the_idx]
        the_energy=self.energy_list[the_idx]
        the_p=self.p_list[the_idx]
        the_T=self.T_list[the_idx]
        the_max_rate=self.BandWidth*np.log(1+the_x)
        
#         print("x",the_x,"max_rate",the_max_rate,"R",the_R,"T",the_T,"p",the_p,"energy",the_energy)
        
        self.the_small_energy=the_energy
        
                
        
    
    def get_op_p(self):
        valid=False
#         while (not valid):
#             self.update_var()
#             self.validate_var()
        # 进行迭代
        for i in range(self.frame_num):
            self.compute_f()
            self.update_x()
            self.update_var()
            if np.isnan(self.delta_x).sum():
#                 print("break1")
                break
            if np.isnan(self.delta_R).sum():
#                 print("break2")
                break
            # 用来记录每一次迭代的结果
            self.return_energy()
            self.validate_x()
        # self.plot_image()
        self.find_the_small()
    

class op_group_energy:
    def __init__(self,D_n):
        self.num_node=D_n.shape[0]
        # 最大节点数，准备改写成分簇函数
        self.child_num_node=3
        self.D_n=D_n
        sort_D_n=np.sort(self.D_n)
        # 分簇
        sort=np.array([[0,3,6],[1,4,7],[2,5,8]])
        self.sum_energy=0
        for i in range(int(self.num_node/self.child_num_node)):
            self.child_D_n=sort_D_n[sort[i,:]]
            op_energy=op_p(num_node=self.child_num_node,D_n=self.child_D_n,SIC=[0,1,2])
            self.sum_energy=self.sum_energy+op_energy.the_small_energy
        print("self.sum_energy",self.sum_energy)
       
            
        
#     def child_group_energy(self):
# #         SIC_list=list(permutations(range(self.child_num_node),self.child_num_node))
# #         all_energy_list=[]
# #         for data in SIC_list:
# #             op_energy=op_p(num_node=self.child_num_node,D_n=self.child_D_n,SIC=data)
# #             all_energy_list.append(op_energy.the_small_energy)
# # #         print("************************all_energy_list**********************************",all_energy_list)
# #         print("smallest_energy",min(all_energy_list))
        
#         return min(all_energy_list)
        
        
        
            
                
        
        
if __name__=="__main__":
#     os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"
    np.random.seed(5)
    # 距离
    D_n=np.random.randint(1*10**6,3*10**6,9)
    group_small_energy=op_group_energy(D_n=D_n)

#     np.random.seed(5)
#     num_node=3
#     D_n=np.array([0.1,0.9,0.3])*1000*10**3
#     SIC_list=list(permutations(range(num_node),num_node))
#     print("SIC_list",SIC_list) #索引表示顺序，值表示节点 比方第四个节点是第一个被解码的,暂时设它信道增益最大
#     all_energy_list=[]
#     print("all_energy_list",all_energy_list)
#     for data in SIC_list:
#         op_energy=op_p(num_node=num_node,D_n=D_n,SIC=data)
#         all_energy_list.append(op_energy.the_small_energy)
#     print("all_energy_list",all_energy_list)
#     print("smallest_energy",min(all_energy_list))
        
    
    
