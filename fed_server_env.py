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
from access import Sat_to_iot
import math
import argparse
from datetime import datetime, timedelta
import os
import copy

def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-dt', '--data', type=str, default='mnist', help='data type')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=20, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.2, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
    parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=2000, help='number of communications')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=int, default=False, help='the way to allocate data to clients')
    parser.add_argument('-isc', '--iscontinue', type=int, default=0, help='if use some model continue')
    parser.add_argument('-moment', '--momentum', type=int, default=0.9, help='momentum')
    parser.add_argument('-sg', '--satellite_group', type=list, default=[[0,1,2,3],[4,5,6,7],[8,9,10,11]], help='the satellite constellation group')
    return parser.parse_args().__dict__

class Fed_server(object):
    def __init__(self, rng, method='default'):
        self.method = method
        self.rng = rng
        self.done = False
        self.args = init_args()
        self.init_global_model()
        self.init_train_args()
        self.accu_dict = []
        self.ene_dict = []
        self.Sat_to_iot= Sat_to_iot(self.args['num_of_clients'], rng)
        self.loss_info = np.zeros(self.args['num_of_clients'])
        self.record = []
        self.accu_record = []
        self.ene_record = []
        self.num_in_comm = int(max(self.args['num_of_clients'] * self.args['cfraction'], 1))
        parent_dir = os.path.abspath(os.getcwd())
        self.dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.method,self.args['data'], 
                                                                    self.args['num_comm'], 
                                                                    self.args['num_of_clients'], 
                                                                    self.args['cfraction'], 
                                                                    self.args['momentum'], 
                                                                    self.args['learning_rate'], 
                                                                    self.args['batchsize'],
                                                                    self.args['epoch'],
                                                                    self.args['IID'])
        path = os.path.join(parent_dir, 'res', self.dir_name)
        if not os.path.exists(path):
            os.mkdir(path)
        
        # 先初始化一下
        with torch.no_grad():
            self.net.load_state_dict(self.global_parameters, strict=True)
            num_batches = len(self.testDataLoader)
            size = len(self.testDataLoader.dataset)
            test_loss, correct = 0, 0
            for data, label in self.testDataLoader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                test_loss += self.loss_func(preds, label).item()
                correct += (preds.argmax(1) == label).type(torch.float).sum().item()
                # preds = torch.argmax(preds, dim=1)
                # sum_accu += (preds == label).float().mean()
                # num += 1
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        self.last_accu = correct
        self.last_loss = test_loss
        self.init_loss = test_loss
        
        self.sum_exp = np.zeros(self.args['num_of_clients'])
        self.exp = np.ones(self.args['num_of_clients'])*0.5
        self.total_num = np.ones(self.args['num_of_clients'], dtype=int)
        
        self.test_local_val = None
        
        
    def get_env_info(self):
        self.Sat_to_iot.out_geometry()
        self.Sat_to_iot.out_h()
        
        local_val_list = [0,0,0,0,0,0,0,0,0,0]
        all_clients = [ 'client{}'.format(i) for i in list(range(10))]
        for index, client in enumerate(all_clients):
            local_value = self.myClients.clients_set[client].local_val(self.net, self.global_parameters,self.loss_func)
            local_val_list[index] = local_value
        
       
        
        # 没有用，应该算选了该设备后，在全局模型上下降了多少loss作为贡献度，用loss除以选拔过的次数，没有选择过该设备的话，所有都设置为1，表示都有机会选择，这一步放到get_reward中去进行更新,设置为self.exp
        
        # if self.test_local_val == None:
        #     self.test_local_val = [i - self.last_loss for i in local_val_list]
        # else:
        #     temp = [i - self.last_loss for i in local_val_list]
        #     self.test_local_val = [math.fabs((self.test_local_val[index]) + math.fabs(temp[index])) for index, i in enumerate(local_val_list)]
        
        # target_list = [ item/(round+1) for index, item in enumerate(self.test_local_val)]
        
        # 映射一下local_val_list
        indices = np.argsort(local_val_list)
        for i, item in enumerate(indices):
            # local_val_list[item] = i*0.1
            # if local_val_list[item]>2:
            #     local_val_list[item] = 1
            # elif local_val_list[item]<0.6:
            #     local_val_list[item] = 0.1
            # # 
            # else:
            #     local_val_list[item] = 
            local_val_list[item] = local_val_list[item]/self.init_loss
                
            
        
        for index in list(range(self.args['num_of_clients'])):
            client_index = index%10
            self.loss_info[index] = local_val_list[client_index]
        
        # 映射一下h_k
        local_hk_list =  [0]*self.args['num_of_clients']
        temp_hk_list = self.Sat_to_iot.h_k_los_value
        indices = np.argsort(temp_hk_list)
        for i, item in enumerate(indices):
            local_hk_list[item] = i*0.01
        # standard_h = 10 * np.ones(self.args['num_of_clients'])
        # normal_accu_info = self.accu_info/standard_accu
        # self.loss_info = (self.loss_info-np.min(self.loss_info))/(np.max(self.loss_info)-np.min(self.loss_info))
        # normal_energy_info = self.Sat_to_iot.h_k_los_value/standard_h
        
        # self.env_info = (np.vstack((normal_accu_info,normal_energy_info)).flatten('F'))*10
        # 先不算模型训练部分的 
        # self.env_info = (np.vstack((local_hk_list)).flatten('F'))*10
        # 先不算通信部分的 
        # self.env_info = (np.vstack((self.loss_info)).flatten('F'))
        self.env_info = (np.vstack((self.loss_info,local_hk_list)).flatten('F'))
        
        return np.expand_dims(self.env_info, axis=0)
    
    def get_reward(self,round):
        self.delta_accu = self.global_accu - self.last_accu
        self.delta_loss = self.last_loss - self.loss 
        if self.last_loss >0.5:
            accu_reward = (self.delta_loss  - self.last_loss/1.5)
        else:
            accu_reward = (self.delta_loss - 0.1)
            
        # if self.last_accu>=0.5:
        #     accu_reward = self.delta_loss - self.last_loss/2
        # else:
        #     accu_reward = ((self.delta_loss - 0.1))
        compute_energy_reward=len(self.order)/50
        
        trans_enrgy_reward = (self.Sat_to_iot.total_E)*np.log2(1+ np.average(self.Sat_to_iot.h_k_los_value)**2)/10
        
        self.reward = accu_reward - compute_energy_reward - trans_enrgy_reward
        self.reward*=10
        
        
        # 实验一下这个方式
        for i in self.order:
            self.sum_exp[i] += accu_reward/10
            self.total_num [i]+=1
        self.exp = self.sum_exp/self.total_num+0.5
        
        
        # if len(self.order) == 0:
        #     self.reward = -10
        self.last_accu =self.global_accu
        self.last_loss =self.loss
        # if len(self.order) ==0:
        #     self.reward = -1000
        # # elif len(self.order) >=50:
        # #     self.reward = -1000
        # else:
        #     self.reward*=10
        # if round< 10:
        #     a = 2.25209022319399
        #     b = 0.0703665527500099
        #     c = -0.0725867426513563
        #     d = 0.00518770741346589
        #     accu_reward = (10*((a+b*round+c*round**2+d*round**3)-self.loss))
        # else:
        #     a = 1.03985939537506
        #     b = -0.0422685414545357
        #     c = 0.000976547034379296
        #     d = -7.95166620691345*10**(-6)
        #     accu_reward = (10*(self.loss-(a+b*round+c*round**2+d*round**3)-self.loss))
        
        # A = 0.786254763206445
        # B = 0.641618966371729
        # C = 1.5164626038415
        # D = 0.100341908272168
        # accu_reward = (((((A - D) / (1 + (round/C)**B)) + D))-self.loss)/1
        # energy_reward=len(self.order)/50
        # self.reward = accu_reward - energy_reward
        # self.reward*=10
            
        # standard_ener, standard_accu = 0, 0.6
        # reward = standard_ener-self.Sat_to_iot.total_E-(standard_accu- self.global_accu)
        # reward = (standard_ener-self.Sat_to_iot.total_E)*np.log2(1+ np.average(self.Sat_to_iot.h_k_los_value)**2)
        # reward = -(standard_accu- self.global_accu)
        # reward = (self.global_accu-1)*np.exp(round)
        # if len(self.order) == 0:
        #     reward = -10
        # reward = reward*100
        # return reward
    
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
        self.myClients = ClientsGroup(self.args['data'], self.args['IID'], 10, self.dev, self.rng)
        self.testDataLoader = self.myClients.test_data_loader

        self.global_parameters = {}
        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()
        
    def reset(self):
        ## sat reset
        self.done = False
        sate_lo=25.01   
        sate_la=0
        self.Sat_to_iot.sate_loca=np.array([sate_lo,sate_la])
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
        # clients_in_comm = ['client{}'.format(i%10) for i in self.order]
        print('clients_in_comm', self.order)
        self.avg_train_local(self.order)
        self.test_and_save(round)
        self.Sat_to_iot.get_power(self.order)
        self.get_reward(round)
        
        # if round > 30:
        #     self.done = True
        
        # self.record.append({'env':self.env_info.tolist(), 'action': action.tolist(), 'reward':reward, 'acu': self.global_accu, 'done': self.done})
        # self.record.append({'env':self.env_info.tolist(), 'action': action.tolist(), 'reward':reward, 'done': self.done})
        # self.record.append({'actions':len(self.order.tolist()), 'reward':reward, 'done': self.done})
        # self.record.append({'state':self.env_info.tolist(),'actions':self.order.tolist(), 'reward':self.reward, 'done': self.done}) 
        self.record.append({'actions':len(self.order.tolist()), 'reward':self.reward, 'done': self.done}) 
        with open('./res/{}/record.txt'.format(self.dir_name), 'w+') as f:
            f.write(json.dumps(self.record))
            f.close()
        return self.done, self.reward

    def avg_train_local(self, order):
        local_parameters = {}
        sum_parameters = None
        cur_parameters = None
        local_parameters_list = [{},{},{},{},{},{},{},{},{},{}]
        # 先分出来10个客户端，然后用100个去算，这里的逻辑等会儿去写
        all_clients = [ 'client{}'.format(i) for i in list(range(10))]
        for index, client in enumerate(all_clients):
            cur_parameters = self.myClients.clients_set[client].localUpdate(self.args['epoch'], self.args['batchsize'], self.net,
                                                                         self.loss_func, self.opti, self.global_parameters)
            for key, var in cur_parameters.items():
                    local_parameters_list[index][key] = var.clone()
            # local_parameters_list.append(copy.copy(cur_parameters))
            # index = int(client.replace('client',''))
        for index in order:
            client_index = index%10
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
        with torch.no_grad():
            if (i + 1) % self.args['val_freq'] == 0:
                self.net.load_state_dict(self.global_parameters, strict=True)
                num_batches = len(self.testDataLoader)
                size = len(self.testDataLoader.dataset)
                test_loss, correct = 0, 0
                for data, label in self.testDataLoader:
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    test_loss += self.loss_func(preds, label).item()
                    correct += (preds.argmax(1) == label).type(torch.float).sum().item()
                    # preds = torch.argmax(preds, dim=1)
                    # sum_accu += (preds == label).float().mean()
                    # num += 1
                test_loss /= num_batches
                correct /= size
                print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
                
                
                self.loss = test_loss
                self.global_accu = correct
                # self.global_accu = (sum_accu / num).item()
                # print('accuracy: {}'.format(self.global_accu))
                self.accu_dict.append(self.global_accu)
                self.ene_dict.append(self.Sat_to_iot.total_E)
                
            # if (len(self.accu_dict)>10 and np.std(np.array(self.accu_dict[-11:-1]))<0.007) or  (len(self.accu_dict)>5 and self.accu_dict[-1]<0.10) or (self.accu_dict[-1]>0.65):
            if len(self.accu_dict)>=30:
                self.accu_record.append(self.accu_dict)
                self.ene_record.append(self.ene_dict)
                self.accu_dict = []
                self.ene_dict = []
                self.done = True
                
                with open("./res/{}/acu.txt".format(self.dir_name),  "w+") as file:
                    # file.write(json.dumps({'accu_dict':self.accu_dict, 'ene_dict':sum(self.ene_dict)}))
                    file.write(json.dumps(self.accu_record))
                    
                    file.close()
                with open("./res/{}/ene.txt".format(self.dir_name),  "w+") as file:
                    # file.write(json.dumps({'accu_dict':self.accu_dict, 'ene_dict':sum(self.ene_dict)}))
                    file.write(json.dumps(self.ene_record))
                    
                    file.close()

            
       


if __name__=="__main__":
    rng2 = np.random.default_rng(seed=100)
    args = init_args()
    fed_server_random = Fed_server(rng2)
    for round in list(range(0, args['num_comm'])):
        fed_server_random.get_env_info(round)
        fed_server_random.train_process(np.array([1,0,0,0,0,0,0,0,0]),round)
        fed_server_random.get_reward(round)
        
