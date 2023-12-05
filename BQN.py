from tqdm import tqdm
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 
import json
import numpy as np 

import random 
import argparse
from branch_model import DuelingNetwork, BranchingQNetwork
from fed_server_env import Fed_server 

def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    ## bqn 参数
    parser.add_argument('-es', '--epsilon_start', type=float, default=0.75,  help='epsilon start')
    parser.add_argument('-ef', '--epsilon_final', type=float, default=0.1,  help='epsilon final')
    parser.add_argument('-ed', '--epsilon_decay', type=int, default=1000,  help='epsilon decay')
    parser.add_argument('-ga', '--gamma', type=float, default=0.95,  help='gamma')
    parser.add_argument('-mlr', '--method_learning_rate', type=float, default=0.001,  help='bqn lr')
    parser.add_argument('-tnuf', '--target_net_update_freq', type=int, default=1000,  help='target net update freq')
    parser.add_argument('-ms', '--memory_size', type=int, default=10000,  help='memory_size')
    parser.add_argument('-bs', '--batch_size', type=int, default=128,  help='batch_size')
    parser.add_argument('-ls', '--learning_starts', type=float, default=256,  help='learning_starts')
    parser.add_argument('-mf', '--max_frames', type=float, default=3000,  help='max_frames')
    
    
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-dt', '--data', type=str, default='mnist', help='data type')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=40, help='numer of the clients')
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
    parser.add_argument('-iid', '--IID', type=bool, default=False, help='the way to allocate data to clients')
    parser.add_argument('-isc', '--iscontinue', type=int, default=0, help='if use some model continue')
    parser.add_argument('-moment', '--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-ib', '--isblur', type=str, default=False, help='if blur the dataset')
    parser.add_argument('-sc', '--sample_client', type=int, default=10, help='sample_client')
    parser.add_argument('-rw', '--reward_weight', type=float, default=0.5, help='reward_weight')
    parser.add_argument('-ss', '--set_size', type=int, default=100, help='set_size')
    parser.add_argument('-pn', '--punish', type=float, default=60, help='punish')
    parser.add_argument('-rm', '--reward_method', type=str, default='variable', help='reward_method')
    parser.add_argument('-sen', '--select_num', type=str, default='variable', help='select_num')
    
    parser.add_argument('-enf', '--env_frame', type=int, default=30, help='env_frame')
    parser.add_argument('-bw', '--bandwidth', type=float, default=200*10**6, help='bandwidth')
    parser.add_argument('-ds', '--data_size', type=float, default=30*8*10**3, help='data_size')
    parser.add_argument('-bn', '--band_num', type=int, default=2, help='band_num')
    parser.add_argument('-tm', '--t_max', type=float, default=5, help='t_max')
    return parser.parse_args().__dict__

class BranchingDQN(nn.Module): 

    def __init__(self, obs, ac, n, args, save_path): 

        super().__init__()

        self.q = BranchingQNetwork(obs, ac, n)
        self.target = BranchingQNetwork(obs, ac, n)

        try: 
            self.q.load_state_dict(torch.load(save_path))
            self.target.load_state_dict(torch.load(save_path))
            
        except:
            self.target.load_state_dict(self.q.state_dict())

        self.target_net_update_freq = args['target_net_update_freq']
        self.update_counter = 0

    def get_action(self, x): 

        with torch.no_grad(): 
            # a = self.q(x).max(1)[1]
            out = self.q(x).squeeze(0)
            action = torch.argmax(out, dim = 1)
            # action_ = np.where(choose == 1)[0]
            # np.random.shuffle(action_)
            # action = action_[:2]
            # action = np.argpartition(act.numpy(), -join_num)[-join_num:]
        return action.numpy()

    def update_policy(self, adam, memory, params): 

        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(params['batch_size'])

        states = torch.tensor(b_states).float()
        actions = torch.tensor(b_actions).long().reshape(states.shape[0],-1,1)
        rewards = torch.tensor(b_rewards).float().reshape(-1,1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1,1)

        qvals = self.q(states)

        current_q_values = self.q(states).gather(2, actions).squeeze(-1)

        with torch.no_grad():
            argmax = torch.argmax(self.q(next_states), dim = 2)
            max_next_q_vals = self.target(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            max_next_q_vals = max_next_q_vals.mean(1, keepdim = True)
        expected_q_vals = rewards + max_next_q_vals*0.99*masks
        # print(expected_q_vals[:5])
        loss = F.mse_loss(expected_q_vals, current_q_values)
        print('loss',loss)
        # input(loss)

        # print('\n'*5)
        
        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters(): 
            p.grad.data.clamp_(-1.,1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0: 
            self.update_counter = 0 
            self.target.load_state_dict(self.q.state_dict())


class ExperienceReplayMemory:
    def __init__(self, capacity, dir_name):
        self.capacity = capacity
        try:
            with open(dir_name,  "r") as file:
                self.memory = json.loads(file.read())
                file.close()
        except:
            self.memory = []
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])


        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
    


if __name__=="__main__":
    rng2 = np.random.default_rng(seed=100)
    args = init_args()
    bins = 2
    prop_path = 'selectNum_{}_bqrLr_{}'.format(args['select_num'], args['method_learning_rate'])
    env = Fed_server(rng2, args, prop_path,'BQN')
    dir_name = env.get_path()
    memory = ExperienceReplayMemory(args['memory_size'], "./res/{}/memory.txt".format(dir_name))
    agent = BranchingDQN(2*env.args['num_of_clients'], env.args['num_of_clients'], bins, args, "./res/{}/model.pt".format(dir_name))
    adam = optim.Adam(agent.q.parameters(), lr = args['method_learning_rate']) 


    s = env.get_env_info(0)
    ep_reward = 0. 
    recap = []
    for frame in range(args['max_frames']): 
        # if_memo = False
        frame_num = frame%args['env_frame']
        
        epsilon = args['epsilon_final']+ (args['epsilon_start'] - args['epsilon_final']) * np.exp(-1. * frame / args['epsilon_decay'])
        
        if np.random.random() > epsilon: 
            action = agent.get_action(torch.tensor(s, dtype=torch.float32))
            print('network choose')
        else: 
            # # # 选择最好的10个或者1个
            # # # 希望在h最好的前提下去随机选择20到40个
            # select_num = 4**(np.random.randint(1,3))
            # action = np.zeros(shape = env.args['num_of_clients'], dtype=int)
            # candi_action_indices = np.argsort(env.Sat_to_iot.h_k_los_value)[:select_num].tolist()
            
            # for i in candi_action_indices:
            #     action[i] = 1
            if args['select_num'] == 'constant':
                limit_multi = 2
            elif args['select_num'] == 'variable':
                limit_multi = int(env.args['num_of_clients']/10)
            select_num_list = [5*(i+1) for i in range(limit_multi)]
            # 本轮的选择个数
            select_num = rng2.choice(select_num_list, 1, replace=False)[0]
            # 备选序列
            candi_action_indices =  np.argsort(env.Sat_to_iot.h_list[frame_num])[::-1][:2*select_num].tolist()
            select_indices = rng2.choice(candi_action_indices, select_num, replace=False)
            action = np.zeros(env.args['num_of_clients'])
            action[select_indices] = 1
                

        done, r = env.train_process(action, frame)
        ns = env.get_env_info(frame+1)
        recap.append(r)  
        # with open('test-bqn.txt','w') as file:
        #         file.write(str(recap))
        #         file.close()
        if done:
            env.reset()
            ns = env.get_env_info(frame+1)
            recap.append(ep_reward)
            ep_reward = 0.
            path = env.get_path()
            torch.save(agent.q.state_dict(), "./res/{}/model.pt".format(path))
            
            # # 存储学习记忆
            # save_memory = []
            # for item in memory.memory:
            #         save_memory.append(item.tolist())
            with open("./res/{}/memory.txt".format(path),  "w+") as file:
                
                    file.write(json.dumps(memory.memory))
                    file.close() 
               
        # if if_memo:
        memory.push((s.reshape(-1).tolist(), action.tolist(), r, ns.reshape(-1).tolist(), 0. if done else 1.))
        s = ns  

        if len(memory.memory) > args['learning_starts']:
            agent.update_policy(adam, memory, args)



