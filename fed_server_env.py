### 有一个调用客户端决策的方法，有一个确定所有参数的方法，有一个更新参数总模型的方法
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN, Cifar_CNN, Cifar_customize
from clients import ClientsGroup, client
import json
from access_env import Sat_to_iot
import math
import argparse
from datetime import datetime, timedelta
import os
import copy
from noma2oma import NOMA_method, OMA_method

def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-dt', '--data', type=str, default='mnist', help='data type')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=20, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.2, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
    parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.001, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=10, help='number of communications')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=int, default=False, help='the way to allocate data to clients')
    parser.add_argument('-isc', '--iscontinue', type=int, default=0, help='if use some model continue')
    parser.add_argument('-moment', '--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-ib', '--isblur', type=str, default=False, help='if blur the dataset')
    parser.add_argument('-sc', '--sample_client', type=int, default=10, help='sample_client')
    parser.add_argument('-rw', '--reward_weight', type=float, default=0.5, help='reward_weight')
    parser.add_argument('-ss', '--set_size', type=int, default=100, help='set_size')
    parser.add_argument('-pn', '--punish', type=float, default=30, help='punish')
    
    parser.add_argument('-ef', '--env_frame', type=int, default=30, help='env_frame')
    parser.add_argument('-bw', '--bandwidth', type=float, default=10**6, help='bandwidth')
    parser.add_argument('-ds', '--data_size', type=float, default=1*8*10**3, help='data_size')
    parser.add_argument('-bn', '--band_num', type=int, default=2, help='band_num')
    parser.add_argument('-tm', '--t_max', type=float, default=1, help='t_max')

    return parser.parse_args().__dict__

class Fed_server(object):
    def __init__(self, rng, prop_args,prop_path, method='default'):
        self.method = method
        self.rng = rng
        self.done = False
        self.args = prop_args
        self.init_global_model()
        self.init_train_args()
        self.accu_dict = []
        self.ene_dict = []
        self.loss_dict = []
        self.Sat_to_iot= Sat_to_iot(self.args['num_of_clients'], rng, self.args['env_frame'])
        
        parent_dir = os.path.abspath(os.getcwd())
        self.prop_path = prop_path
        self.dir_name = 'method_{}/clientNum_{}/iid_{}/data_{}_envFrame_{}_rewardWeight_{}_isblur_{}_sampleNum_{}_setSize_{}_punish_{}_rewardMethod_{}_{}'.format(self.method, self.args['num_of_clients'], self.args['IID'],
                                                                    self.args['data'],self.args['env_frame'],self.args['reward_weight'],self.args['isblur'],self.args['sample_client'],self.args['set_size'],self.args['punish'],self.args['reward_method'],
                                                                    self.prop_path)
        self.path = os.path.join(parent_dir, 'res', self.dir_name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
        try:
            with open("./res/{}/acu.txt".format(self.dir_name),  "r") as file:
                self.accu_record = json.loads(file.read())
                file.close()
            with open("./res/{}/loss.txt".format(self.dir_name),  "r") as file:
                self.loss_record = json.loads(file.read())
                file.close()
            with open("./res/{}/ene.txt".format(self.dir_name),  "r") as file:
                self.ene_record = json.loads(file.read())
                file.close()
            with open("./res/{}/record.txt".format(self.dir_name),  "r") as file:
                self.record = json.loads(file.read())
                file.close()
        except:
            self.accu_record = []
            self.ene_record = []
            self.loss_record = []
            self.record = []
            
        self.local_accu_list = np.zeros(self.args['sample_client'])
        self.local_loss_list = np.zeros(self.args['sample_client'])
        self.loss_info = np.zeros(self.args['num_of_clients'])
        self.h_info =  np.zeros(self.args['num_of_clients'])
        
    def get_path(self):
        return self.dir_name
    
    def get_env_info(self, round):
        frame_num = round%self.args['env_frame']
        # 如果是第一幕，应该算一下
        if frame_num == 0:
            all_clients = [ 'client{}'.format(i) for i in range(self.args['sample_client'])]
            for index, client in enumerate(all_clients):
                local_accu, local_loss = self.myClients.clients_set[client].local_val(self.net, self.global_parameters,self.loss_func)
                self.local_loss_list[index] = local_loss
                self.local_accu_list[index] = local_accu
            self.avg_local_loss = np.average(self.local_loss_list)
            self.avg_local_accu = np.average(self.local_accu_list)
            # 第一幕的env中做最初讲
            self.last_accu = self.avg_local_accu
            self.last_loss = self.avg_local_loss
            
            self.loss_dict.append(self.avg_local_loss)
            self.accu_dict.append(self.avg_local_accu)
        
        # 排列映射
        indices = np.argsort(self.local_loss_list)
        map_local_loss = np.zeros(self.args['sample_client'])
        for index, item in enumerate(indices):
            map_local_loss[item] = index/self.args['sample_client']
              
        # 扩展到多个客户端上
        for index in range(self.args['num_of_clients']):
            client_index = index%10
            self.loss_info[index] = map_local_loss[client_index]
        
        # 映射一下h_k
        
        map_local_hk = self.Sat_to_iot.h_list[frame_num]
        # 分为10个等级
        min_hk, max_hk = np.min(map_local_hk), np.max(map_local_hk)
        hk_range = max_hk - min_hk
        for i, temp_h_k in enumerate(map_local_hk):
            level = ((temp_h_k - min_hk)/hk_range)
            self.h_info[i] = level
        
        
        # indices = np.argsort(temp_hk_list)
        # for i, item in enumerate(indices):
        #     local_hk_list[item] = i*0.01
        # standard_h = 10 * np.ones(self.args['num_of_clients'])
        # normal_accu_info = self.accu_info/standard_accu
        # self.loss_info = (self.loss_info-np.min(self.loss_info))/(np.max(self.loss_info)-np.min(self.loss_info))
        # normal_energy_info = self.Sat_to_iot.h_k_los_value/standard_h
        
        # self.env_info = (np.vstack((normal_accu_info,normal_energy_info)).flatten('F'))*10
        # 先不算模型训练部分的 
        # self.env_info = (np.vstack((local_hk_list)).flatten('F'))
        # 先不算通信部分的 
        # self.env_info = (np.vstack((self.loss_info)).flatten('F'))
        self.env_info = (np.vstack((self.loss_info,self.h_info)).flatten('F'))
        
        return np.expand_dims(self.env_info, axis=0)
    
    def get_reward(self,round):
        frame_num = round%self.args['env_frame']
        with torch.no_grad():
            all_clients = [ 'client{}'.format(i) for i in range(self.args['sample_client'])]
            for index, client in enumerate(all_clients):
                local_accu, local_loss = self.myClients.clients_set[client].local_val(self.net, self.global_parameters,self.loss_func)
                self.local_loss_list[index] = local_loss
                self.local_accu_list[index] = local_accu
                
        self.avg_local_loss = np.average(self.local_loss_list)
        self.avg_local_accu = np.average(self.local_accu_list)
        
        # 存储dict
        self.loss_dict.append(self.avg_local_loss)
        self.accu_dict.append(self.avg_local_accu)
        
        self.delta_accu = self.avg_local_accu - self.last_accu
        self.delta_loss = self.last_loss - self.avg_local_loss
        
        
        if self.delta_loss >0:
            if self.args['reward_method'] == 'constant':
                self.loss_reward = 1
            elif self.args['reward_method'] == 'variable':
                self.loss_reward = (self.delta_loss  - self.last_loss/2)
        else:
            self.loss_reward = -1*self.args['punish']
        # if self.last_loss >0.5:
        #     self.loss_reward = (self.delta_loss  - self.last_loss/1.5)*2
        # else:
        #     self.loss_reward = (self.delta_loss - 0.1)*2
        
        # 更新last loss 和 accu 值
        self.last_accu = self.avg_local_accu
        self.last_loss = self.avg_local_loss
        # if self.last_accu>=0.5:
        #     accu_reward = self.delta_loss - self.last_loss/2
        # else:
        #     accu_reward = ((self.delta_loss - 0.1))
        self.compute_energy_reward=len(self.order)/20
        if len(self.order) > 0:
            
            self.noma_output_energy =  NOMA_method(self.Sat_to_iot.h_list[frame_num][self.order], self.args['bandwidth'], self.args['data_size'], self.args['band_num'], self.args['t_max']).output_energy
            self.oma_output_energy = OMA_method(self.Sat_to_iot.h_list[frame_num][self.order], self.args['bandwidth'], self.args['data_size'], self.args['t_max']).output_energy
            self.trans_enrgy_reward = self.noma_output_energy*100
        else:
            self.noma_output_energy = 0
            self.oma_output_energy = 0
            self.trans_enrgy_reward = 0
        self.reward = self.args['reward_weight']*self.loss_reward*1 - (1-self.args['reward_weight'])*(self.compute_energy_reward + self.trans_enrgy_reward)*1
        
        # 存储 dict
        self.ene_dict.append({'noma':self.noma_output_energy,'oma':self.oma_output_energy,'loss_reward':self.loss_reward,'trans_reward':self.trans_enrgy_reward, 'comp_reward':self.compute_energy_reward})
        # self.reward = - trans_enrgy_reward
        # if len(self.order) == 0:
        #     self.reward = -20
        
        
      
    
    def init_global_model(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args['gpu']
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = None
        if self.args['model_name'] == 'mnist_2nn':
            net = Mnist_2NN()
        elif self.args['model_name'] == 'mnist_cnn':
            net = Mnist_CNN()
        elif self.args['model_name'] == 'cifar_cnn':
            net = Cifar_CNN()
        elif self.args['model_name'] == 'cifar_costu':
            net = Cifar_customize()
        self.net = net.to(self.dev)
    
    def init_train_args(self):
        self.loss_func = F.cross_entropy
        self.opti = optim.SGD(self.net.parameters(), lr=self.args['learning_rate'], momentum = self.args['momentum'])
        # self.opti = optim.SGD(self.net.parameters(), lr=self.args['learning_rate'])
        self.myClients = ClientsGroup(self.args['data'], self.args['IID'], self.args['sample_client'], self.dev, self.rng, self.args['isblur'], self.args['set_size'])
        self.testDataLoader = self.myClients.test_data_loader

        self.global_parameters = {}
        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()
        
    def reset(self):
        ## sat reset
        self.done = False
        # fed reset
        self.accu_dict = []
        self.ene_dict = []
        self.init_global_model()
        self.global_parameters = {}
        self.loss_func = F.cross_entropy
        self.opti = optim.SGD(self.net.parameters(), lr=self.args['learning_rate'], momentum = self.args['momentum'])
        # self.opti = optim.SGD(self.net.parameters(), lr=self.args['learning_rate'])
        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()
        self.accu_info = np.zeros(self.args['num_of_clients'])
        


    def train_process(self, action, round):
        self.order = np.where(action == 1)[0]
        np.sort(self.order)
        clients_in_comm = ['client{}'.format(i%10) for i in self.order]
        print('clients_in_comm', self.order)
        self.avg_train_local(self.order)
        self.get_reward(round)
        self.test_and_save(round)
    
        
        self.record.append({'actions':len(self.order.tolist()), 'reward':self.reward, 'done': self.done}) 
        with open('./res/{}/record.txt'.format(self.dir_name), 'w+') as f:
                f.write(json.dumps(self.record))
                f.close() 
        
        return self.done, self.reward

    def avg_train_local(self, order):
        local_parameters = {}
        sum_parameters = None
        cur_parameters = None
        local_parameters_list = []
        # 先分出来10个客户端，然后用100个去算，这里的逻辑等会儿去写
        all_clients = [ 'client{}'.format(i) for i in range(self.args['sample_client'])]
        for index, client in enumerate(all_clients):
            local_parameters_list.append({})
            cur_parameters = self.myClients.clients_set[client].localUpdate(self.args['epoch'], self.args['batchsize'], self.net,
                                                                         self.loss_func, self.opti, self.global_parameters)
            for key, var in cur_parameters.items():
                local_parameters_list[index][key] = var.clone()
                    
            # local_parameters_list.append(copy.copy(cur_parameters))
            # index = int(client.replace('client',''))
        for index in order:
            client_index = index%self.args['sample_client']
            local_parameters = local_parameters_list[client_index]
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        try:
            for var in self.global_parameters:
                self.global_parameters[var] = (sum_parameters[var] / len(order))
        except:
            self.global_parameters = self.global_parameters

    
    def test_and_save(self, i):
            # 存储信息
            
            
            # if (len(self.accu_dict)>10 and np.std(np.array(self.accu_dict[-11:-1]))<0.007) or  (len(self.accu_dict)>5 and self.accu_dict[-1]<0.10) or (self.accu_dict[-1]>0.65):
            if len(self.ene_dict)==self.args['env_frame']:
                self.accu_record.append(self.accu_dict)
                self.ene_record.append(self.ene_dict)
                self.loss_record.append(self.loss_dict)
                
                self.accu_dict = []
                self.ene_dict = []
                self.loss_dict = []
                self.done = True
                
                with open("./res/{}/acu.txt".format(self.dir_name),  "w+") as file:
                    # file.write(json.dumps({'accu_dict':self.accu_dict, 'ene_dict':sum(self.ene_dict)}))
                    file.write(json.dumps(self.accu_record))
                    
                    file.close()
                with open("./res/{}/ene.txt".format(self.dir_name),  "w+") as file:
                    # file.write(json.dumps({'accu_dict':self.accu_dict, 'ene_dict':sum(self.ene_dict)}))
                    file.write(json.dumps(self.ene_record))
                    
                    file.close()
                
                with open("./res/{}/loss.txt".format(self.dir_name),  "w+") as file:
                    # file.write(json.dumps({'accu_dict':self.accu_dict, 'ene_dict':sum(self.ene_dict)}))
                    file.write(json.dumps(self.loss_record))
                    
                    file.close()   
                

            
       


if __name__=="__main__":
    rng2 = np.random.default_rng(seed=100)
    args = init_args()
    fed_server_random = Fed_server(rng2,args,'')
    for round in range(30):
        fed_server_random.get_env_info(round)
        fed_server_random.train_process(np.array([1]*20),round)
        if fed_server_random.done == True:
            fed_server_random.reset()
        
