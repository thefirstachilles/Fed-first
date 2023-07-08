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

def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-dt', '--data', type=str, default='cifar', help='data type')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=10, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.2, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
    parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='cifar_costu', help='the model to train')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.001, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=2000, help='number of communications')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=int, default=True, help='the way to allocate data to clients')
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
        self.test_and_save(0)
        self.accu_info = self.global_accu*np.ones(self.args['num_of_clients'])
        self.record = []

    def get_env_info(self):
        self.Sat_to_iot.out_geometry()
        self.Sat_to_iot.out_h()
        self.Sat_to_iot.get_power()
        standard_accu = 10 * np.ones(self.args['num_of_clients'])
        standard_ene = 0.2 * np.ones(self.args['num_of_clients'])
        # standard_accu_info = standard_accu/np.linalg.norm(standard_accu)
        # standard_ene_info = standard_ene/np.linalg.norm(standard_ene)/standard_accu_info
        normal_accu_info = self.accu_info/standard_accu
        normal_energy_info = self.Sat_to_iot.E_client/standard_ene
        self.env_info = (np.vstack((normal_accu_info,normal_energy_info)).flatten('F'))*10
        return np.expand_dims(self.env_info, axis=0)
    
    def get_reward(self):
        standard_ener, standard_accu = 0, 0.6
        reward = standard_ener-self.Sat_to_iot.total_E-(standard_accu- self.global_accu)
        if len(self.order) == 0:
            reward = -1
        reward = reward*10
        return reward
    
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

        self.myClients = ClientsGroup(self.args['data'], self.args['IID'], self.args['num_of_clients'], self.dev, self.rng)
        self.testDataLoader = self.myClients.test_data_loader
        self.num_in_comm = int(max(self.args['num_of_clients'] * self.args['cfraction'], 1))

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
        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()
        self.test_and_save(0)
        self.accu_info = self.global_accu*np.ones(self.args['num_of_clients'])
        


    def train_process(self, action, round):
        # order = rng2.permutation(self.args['num_of_clients'])]
        self.order = np.where(action == 1)[0]
        np.random.shuffle(self.order)
        clients_in_comm = ['client{}'.format(i) for i in self.order]
        print('clients_in_comm', clients_in_comm)
        self.avg_train_local(clients_in_comm)
        self.test_and_save(round)
        reward = self.get_reward()
        self.record.append({'env':self.env_info.tolist(), 'action': action.tolist(), 'reward':reward, 'acu': self.global_accu, 'done': self.done})
        with open('./res/{}_record.txt'.format(self.method),'w') as f:
            f.write(json.dumps(self.record))
            f.close()
        return self.done, reward

    def avg_train_local(self, clients_in_comm):
        local_parameters = {}
        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters, accu = self.myClients.clients_set[client].localUpdate(self.args['epoch'], self.args['batchsize'], self.net,
                                                                         self.loss_func, self.opti, self.global_parameters)
            
            index = int(client.replace('client',''))
            self.accu_info[index] = accu.item()
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        try:
            for var in self.global_parameters:
                self.global_parameters[var] = (sum_parameters[var] / len(clients_in_comm))
        except:
            self.global_parameters = self.global_parameters

    
    def test_and_save(self, i):
        with torch.no_grad():
            if (i + 1) % self.args['val_freq'] == 0:
                self.net.load_state_dict(self.global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in self.testDataLoader:
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                self.global_accu = (sum_accu / num).item()
                print('accuracy: {}'.format(self.global_accu))
                self.accu_dict.append(self.global_accu)
                self.ene_dict.append(self.Sat_to_iot.total_E)
            if (len(self.accu_dict)>10 and np.std(np.array(self.accu_dict[-11:-1]))<0.007) or  (len(self.accu_dict)>5 and self.accu_dict[-1]<0.10) or (self.accu_dict[-1]>0.65):
                self.done = True
                with open("./res/{}_{}_{}_{}_{}_{}_{}_{}_{}_acu.txt".format(self.method,
                                                                    self.args['data'], 
                                                                    self.args['num_comm'], 
                                                                    self.args['num_of_clients'], 
                                                                    self.args['cfraction'], 
                                                                    self.args['momentum'], 
                                                                    self.args['learning_rate'], 
                                                                    self.args['batchsize'],
                                                                    self.args['epoch'],
                                                                    self.args['IID']),  "a+") as file:
                    file.write(json.dumps({'accu_dict':self.accu_dict, 'ene_dict':sum(self.ene_dict)}))
                    file.close()

            
       


if __name__=="__main__":
    rng2 = np.random.default_rng(seed=100)
    args = init_args()
    fed_server_random = Fed_server(args, rng2)
    for round in list(range(0, args['num_comm'])):
        fed_server_random.get_env_info()
        fed_server_random.train_process(round)
        
