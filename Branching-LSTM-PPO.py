#PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import argparse
import numpy as np
from fed_server_env import Fed_server 

#Hyperparameters
def init_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    ## bqn 参数
    # parser.add_argument('-es', '--epsilon_start', type=float, default=1,  help='epsilon start')
    # parser.add_argument('-ef', '--epsilon_final', type=float, default=0.01,  help='epsilon final')
    # parser.add_argument('-ed', '--epsilon_decay', type=int, default=8000,  help='epsilon decay')
    # parser.add_argument('-ms', '--memory_size', type=int, default=100000,  help='memory_size')
    parser.add_argument('-lpl', '--lstm_ppo_lr', type=float, default=0.0005,  help='learning_rate')
    parser.add_argument('-ga', '--gamma', type=float, default=0.99,  help='gamma')
    parser.add_argument('-lmbda', '--lmbda', type=float, default=0.95,  help='lambda')
    parser.add_argument('-ec', '--eps_clip', type=float, default=0.1,  help='eps_clip')
    parser.add_argument('-ke', '--k_epoch', type=int, default=2,  help='k_epoch')
    parser.add_argument('-Th', '--T_horizon', type=int, default=40,  help='T_horizon')
    return parser.parse_args().__dict__
# learning_rate = 0.0005
# gamma         = 0.98
# lmbda         = 0.95
# eps_clip      = 0.1
# K_epoch       = 2
# T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, obs, ac, n, args):
        super().__init__()
        self.args = args
        self.data = []
        self.fc1   = nn.Linear(obs,64)
        self.lstm  = nn.LSTM(64,32)
        self.fc_pi = nn.ModuleList([nn.Linear(32, n) for i in range(ac)])
        self.fc_v  = nn.Linear(32,1)
        

        self.optimizer = optim.Adam(self.parameters(), lr=self.args['lstm_ppo_lr'])

        self.gamma = args['gamma']
        self.lmbda = args['lmbda']
        self.eps_clip = args['eps_clip']
    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        prob = []
        for i, branch in enumerate(self.fc_pi):
            prob.append(Categorical(logits=self.fc_pi[i](x)))

        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.int64), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(self.args['k_epoch']):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            policy, _ = self.pi(s, first_hidden)
            pi_a = []
            for i, action_branch in enumerate(policy):
                pi_a.append(action_branch.probs.squeeze(1).gather(1, a[:,i].unsqueeze(1)))
            pi_a = torch.stack(pi_a, dim=2).squeeze(1)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a)).mean(1, True) # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
        
def main():
    rng2 = np.random.default_rng(seed=100)
    args = init_args()
    env = Fed_server(rng2, 'Branch-Lstm-PPO')
    bins = 2
    model = PPO(2*env.args['num_of_clients'], env.args['num_of_clients'], bins, args)
    print_interval = 20
    recap = []
    
    for n_epi in range(10000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        s = env.get_env_info()
        done = False
        
        while not done:
            for t in range(args['T_horizon']):
                h_in = h_out
                policy, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                _actions = []
                probs = []
                for action_branch in policy:
                    action = action_branch.sample()
                    _actions.append(action.item())
                    probs.append(action_branch.probs[0,0,action.item()].item())

                done, r = env.train_process(np.array(_actions), t)
                s_prime = env.get_env_info()

                # s_prime, r, done, = env.get_env_info(a)

                model.put_data((s, np.array(_actions), r, s_prime, np.array(probs)[np.array(_actions)], h_in, h_out, done))
                s = s_prime

                recap.append(r)
                # with open('test-branch-lstm-ppo.txt','w') as file:
                #     file.write(str(recap))
                #     file.close()

                if done:
                    env.reset()
                    break
                    
            model.train_net()

        # if n_epi%print_interval==0 and n_epi!=0:
        #     print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        #     score = 0.0


if __name__ == '__main__':
    main()