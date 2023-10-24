import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from fed_server_env import Fed_server 
class Config:
    def __init__(self):
        self.algo_name = 'DDPG'
        self.render_mode = 'rgb_array'
        self.train_eps = 100
        self.test_eps = 10
        self.max_steps = 200
        self.batch_size = 128
        self.memory_capacity = 10000
        # self.lr_a = 2e-4
        self.lr_a = 0.001
        # self.lr_c = 5e-4
        self.lr_c = 0.002
        self.gamma = 0.9
        self.sigma = 0.01
        self.tau = 0.005
        self.seed = random.randint(0, 100)
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.n_states = None
        self.n_actions = None
        self.action_bound = None
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')



class ReplayBuffer:
    def __init__(self, cfg):
        self.buffer = np.empty(cfg.memory_capacity, dtype=object)
        self.size = 0
        self.pointer = 0
        self.capacity = cfg.memory_capacity
        self.batch_size = cfg.batch_size
        self.device = cfg.device

    def push(self, transitions):
        self.buffer[self.pointer] = transitions
        self.size = min(self.size + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def clear(self):
        self.buffer = np.empty(self.capacity, dtype=object)
        self.size = 0
        self.pointer = 0

    def sample(self):
        batch_size = min(self.batch_size, self.size)
        indices = np.random.choice(self.size, batch_size, replace=False)
        samples = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
                                             device=self.device), zip(*self.buffer[indices]))
        return samples


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states, cfg.actor_hidden_dim)
        self.fc2 = nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim)
        self.fc3 = nn.Linear(cfg.actor_hidden_dim, cfg.n_actions)
        self.action_bound = cfg.action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))*self.action_bound + self.action_bound
        return action

class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states + cfg.n_actions, cfg.critic_hidden_dim)
        self.fc2 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc3 = nn.Linear(cfg.critic_hidden_dim, 1)


    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class DDPG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = ReplayBuffer(cfg)
        self.actor = Actor(cfg).to(cfg.device)
        self.actor_target = Actor(cfg).to(cfg.device)
        self.critic = Critic(cfg).to(cfg.device)
        self.critic_target = Critic(cfg).to(cfg.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr_a)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.lr_c)
        self.critic_target.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.cfg.device)
        a = self.actor(state).squeeze(0).cpu().numpy()
        a += self.cfg.sigma * np.random.randn(self.cfg.n_actions)
        
        return a


    def update(self):
        if self.memory.size < self.cfg.batch_size:
            return 0, 0
        states, actions, rewards, next_states, dones = self.memory.sample()
        rewards, dones =  rewards.view(-1, 1), dones.view(-1, 1)
        next_q_value = self.critic_target(next_states.squeeze(1), self.actor_target(next_states).squeeze(1))
        target_q_value = rewards + (1 - dones) * self.cfg.gamma * next_q_value

        critic_loss = torch.mean(F.mse_loss(self.critic(states.squeeze(1), actions.squeeze(1)), target_q_value))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -torch.mean(self.critic(states.squeeze(1), self.actor(states).squeeze(1)))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.update_params()

        return actor_loss.item(), critic_loss.item()

    def update_params(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)


def env_agent_config(cfg):
    rng2 = np.random.default_rng(seed=100)
    env = Fed_server(rng2, 'DDPG')
    cfg.n_states = 2*env.args['num_of_clients']
    # cfg.n_actions = env.action_space.shape[0]
    cfg.n_actions = env.args['num_of_clients']
    cfg.action_bound = 0.5
    agent = DDPG(cfg)
    return env, agent


def train(env, agent, cfg):
    rewards, steps = [], []
    for i in range(cfg.train_eps):
        ep_reward, ep_step = 0.0, 0
        state = env.get_env_info()
        critic_loss, actor_loss = 0.0, 0.0
        for round in range(cfg.max_steps):
            ep_step += 1
            a = agent.choose_action(state)
            action = np.zeros(env.args['num_of_clients'], dtype=int)
            for index, ele in enumerate(a):
                if ele > 0.5:
                    action[index] = 1
            done, reward = env.train_process(action, round)
            next_state = env.get_env_info()
            agent.memory.push((state, a, reward, next_state, done))
            state = next_state
            c_loss, a_loss = agent.update()
            critic_loss += c_loss
            actor_loss += a_loss
            ep_reward += reward
            if done:
                env.reset()
                break
    return rewards, steps


def test(agent, cfg):
    print('开始测试!')
    rewards, steps = [], []
    env = gym.make(cfg.env_name, render_mode='human')
    for i in range(cfg.test_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        for _ in range(cfg.max_steps):
            ep_step += 1
            a = agent.choose_action(state)
            action = []
            for index, ele in enumerate(a):
                if ele > 0.5:
                    action.append(index)
            done, reward = env.train_process(action)
            state = env.get_env_info()
            ep_reward += reward
            if terminated or truncated:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f'回合:{i + 1}/{cfg.test_eps}, 奖励:{ep_reward:.3f}')
    print('结束测试!')
    env.close()
    return rewards, steps


if __name__ == '__main__':
    cfg = Config()
    env, agent = env_agent_config(cfg)
    train_rewards, train_steps = train(env, agent, cfg)
    # test_rewards, test_steps = test(agent, cfg)