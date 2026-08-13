import sys
import copy
import pandas as pd
import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn
import torch.optim as optim

from Logger import Logger
from User import User
from Handover import Handover
from Topology import Topology
from Network import Network

log = Logger('./log/RL/RL.log',level='debug',w_level='info')


def resolve_device(device='auto'):
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
            columns=['state', 'action', 'reward', 'next_state'])    #创建表格
        self.i = 0  #下一个transition写入的索引
        self.count = 0  #memory中的transition数量，最大为capacity
        self.capacity = capacity

    def store(self, *args):     #*args没有利用的多余参数保存为元组，**args保存为字典
        if self.count==self.capacity:
            del self.memory.loc[self.i,'state'][:]
        self.memory.loc[self.i] = args
        self.i = (self.i+1)%self.capacity
        self.count = min(self.count+1,self.capacity)
    
    def sample(self, size):
        indices = np.random.choice(self.count,size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)


class UserAgent:
    def __init__(self, user:User, env:Handover, c_agent, gamma=0.9, epsilon=0.01, batch=1024, hidden_sizes=[72,72], device='auto'):
        self.action_n = env.topo.total_sat
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch
        self.device = resolve_device(device)
        self.evaluate_net = self.build_net(input_size=4*self.action_n + 1,   #第一维的数量，即行数
            hidden_sizes=hidden_sizes, output_size=self.action_n).to(self.device)
        self.user = user
        self.c_agent = c_agent
        self.env = env

    def build_net(self, input_size, hidden_sizes, output_size):
        layers = []
        for input_size, output_size in zip(
                [input_size,] + hidden_sizes, hidden_sizes + [output_size,]):   #[[input_size,hidden_sizes],[hidden_sizes,input_size]]
            layers.append(nn.Linear(input_size, output_size))   #设置全连接层
            layers.append(nn.ReLU())    #线性整流函数（Rectified Linear Unit）,将输入小于0的值幅值为0，输入大于0的值不变。
        layers = layers[:-1]    #从0到-1位置的元素，即去掉最后一个(为什么要去掉最后一个)
        model = nn.Sequential(*layers)
        return model

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []
            self.target_net = copy.deepcopy(self.evaluate_net).to(self.device)

    def step(self,observation,reward):
        if self.mode=='train' and np.random.rand()<self.epsilon:
            selected = []
            for sat in self.user.sat_covered:
                selected.append(sat.ID-1)
            action = np.random.choice(selected)
        else:
            state_tensor = torch.as_tensor(observation,dtype=torch.float, device=self.device).squeeze(0)
            q_tensor = self.evaluate_net(state_tensor)
            for i in range(self.action_n):
                if observation[i]==-1.0:
                    q_tensor[i]=-1.0
                else:
                    q_tensor[i]+=100000.0   #修正q值，保证覆盖卫星的q值大于其他的，避免出错
            action_tensor = torch.argmax(q_tensor)
            action = action_tensor.item()
        if self.mode=='train':
            self.trajectory += [observation, reward, action]
            if len(self.trajectory)>=6:
                state, _, act, next_state, reward, _ = self.trajectory[-6:]
                self.c_agent.replayer.store(state,act,reward,next_state)
        return action
    
    def observe(self,mode:str):
        observation = self.env.Observe(self.user,mode)
        return observation
    
    def get_reward(self):
        reward = self.env.Get_Reward(self.user)
        return reward
    

class CenterAgent:
    def __init__(self, env:Handover, gamma=0.9, epsilon=0.01, batch=1024, buffer=100000, hidden_sizes=[72,72],lr=0.001, device='auto'):
        self.action_n = env.topo.total_sat
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch
        self.device = resolve_device(device)
        self.replayer = DQNReplayer(buffer)
        self.evaluate_net = self.build_net(input_size=4*self.action_n + 1,   #第一维的数量，即行数
            hidden_sizes=hidden_sizes, output_size=self.action_n).to(self.device)
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=lr)   #Adam包括偏置修正，修正从原点初始化的一阶矩（动量项）和（非中心的）二阶矩估计
        self.loss = nn.MSELoss()    #均方损失函数

    def build_net(self, input_size, hidden_sizes, output_size):
        layers = []
        for input_size, output_size in zip(
                [input_size,] + hidden_sizes, hidden_sizes + [output_size,]):   #[[input_size,hidden_sizes],[hidden_sizes,input_size]]
            layers.append(nn.Linear(input_size, output_size))   #设置全连接层
            layers.append(nn.ReLU())    #线性整流函数（Rectified Linear Unit）,将输入小于0的值幅值为0，输入大于0的值不变。
        layers = layers[:-1]    #从0到-1位置的元素，即去掉最后一个(为什么要去掉最后一个)
        model = nn.Sequential(*layers)
        return model

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []
            self.target_net = copy.deepcopy(self.evaluate_net).to(self.device)
    
    def learn(self):
        # replay
        states, actions, rewards, next_states = self.replayer.sample(self.batch_size) # replay transitions
        state_tensor = torch.as_tensor(states, dtype=torch.float, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float, device=self.device)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float, device=self.device)

        # train
        next_q_tensor = self.target_net(next_state_tensor)
        next_max_q_tensor, _ = next_q_tensor.max(axis=-1)
        target_tensor = reward_tensor + self.gamma * next_max_q_tensor
        pred_tensor = self.evaluate_net(state_tensor)
        q_tensor = pred_tensor.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        loss_tensor = self.loss(target_tensor, q_tensor)
        self.optimizer.zero_grad()  #optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
        loss_tensor.backward()  #当完成计算后通过调用 .backward(),自动计算所有的梯度
        self.optimizer.step()
    
    def save_net(self, PATH):
        torch.save(self.evaluate_net, PATH)

def train_episode(env:Handover, u_agents, c_agent, model, mode=None,start_time=0, end_time=250000, time_step=50, net_step=10):
    total_reward = 0.0  #训练阶段总回报
    time = start_time   #当前训练时刻
    episode = 0            #训练次数
    actions = {}        #每个时隙内的agent动作集
    env.reset(time,'NETWORK_LOAD')  #
    c_agent.reset(mode=mode)
    temp_reward = 0
    ob_re = {}  #{agent:[observation,reward]}
    reward_list = []
    for agent in u_agents:
        agent.reset(mode=mode)
        ob_re[agent] = [agent.observe('NETWORK_LOAD'),0.0]
    while(time<=end_time):
        episode_reward = 0.0
        for agent in u_agents:
            ob_re[agent][0] = agent.observe('NETWORK_LOAD')
            action = agent.step(ob_re[agent][0],ob_re[agent][1])
            actions[agent.user]= action+1
            episode_reward += ob_re[agent][1]
        #c_agent.update_transition()
        total_reward += episode_reward
        temp_reward += episode_reward
        if c_agent.replayer.count>=0.5*c_agent.replayer.capacity:
            c_agent.learn()
            if(episode%net_step==0):
                c_agent.target_net = copy.deepcopy(c_agent.evaluate_net).to(c_agent.device)
            for agent in u_agents:
                agent.evaluate_net = copy.deepcopy(c_agent.evaluate_net).to(agent.device)
        if episode==0:
            env.step(actions,'INITIAL')
        else:
            env.step(actions,'NETWORK')
        log.logger.debug('time %d: reward = %.4f, episodes = %d',
            time, episode_reward, episode)
        if episode%400==399:
            average_reward = temp_reward/400.0
            log.logger.info('average_reward for past %d average_reward = %.4f, episodes = %d',
                400, average_reward, episode)
            temp_reward = 0
        
        time+=time_step
        episode+=1
        actions.clear()  
        for agent in u_agents:
            ob_re[agent][1] = agent.get_reward()
        env.Update_Env(time,'NETWORK_LOAD')
    log.logger.warning('from %d to %d by %d: average_reward = %.4f, episodes = %d',
        start_time, end_time, time_step, total_reward/episode, episode)
    c_agent.save_net('./log/model/%s.pkl'%model)
    env.close()
    return episode_reward

def play_episode(env:Handover, u_agents, c_agent, model='model1', mode=None,start_time=0, end_time=10000, time_step=30):
    total_reward = 0.0
    time = start_time
    episode = 0
    actions = {}
    env.reset(time,'NETWORK_LOAD')
    ob_re = {}
    model = './log/model/%s.pkl'%model
    log.logger.debug("Loading %s", model)
    for agent in u_agents:
        agent.reset()
        agent.evaluate_net = torch.load(model, map_location=agent.device).to(agent.device)
        ob_re[agent] = [agent.observe('NETWORK_LOAD'),0.0]
    while(time<=end_time):
        episode_reward = 0.0
        for agent in u_agents:
            ob_re[agent][0] = agent.observe('NETWORK_LOAD')
            action = agent.step(ob_re[agent][0],ob_re[agent][1])
            actions[agent.user]= action+1
            episode_reward += ob_re[agent][1]
        #c_agent.update_transition()
        total_reward += episode_reward
        if episode==0:
            env.step(actions,'INITIAL')
        else:
            env.step(actions,'NETWORK')
        log.logger.debug('time %d: reward = %.4f, steps = %d',
            time, episode_reward, episode)
        time+=time_step
        episode+=1
        actions.clear()  
        for agent in u_agents:
            ob_re[agent][1] = agent.get_reward()
        env.Update_Env(time,'NETWORK_LOAD')
    log.logger.info('from %d to %d by %d: average_reward = %.4f, episodes = %d',
        start_time, end_time, time_step, total_reward/episode, episode)
    env.close()
    return episode_reward

def play_episode_random(env:Handover, u_agents, mode=None,start_time=0, end_time=5000, time_step=50):
    total_reward = 0.0
    time = start_time
    episode = 0
    actions = {}
    env.reset(time,'NETWORK_LOAD')
    ob_re = {}
    for agent in u_agents:
        ob_re[agent] = [agent.observe('NETWORK_LOAD'),0.0]
    while(time<=end_time):
        episode_reward = 0.0
        for agent in u_agents:
            ob_re[agent][0] = agent.observe('NETWORK_LOAD')
            selected = [i for i in range(agent.action_n) if ob_re[agent][0][i]>-1.0]
            action = np.random.choice(selected)
            actions[agent.user]= action+1
            episode_reward += ob_re[agent][1]
        total_reward += episode_reward
        if episode==0:
            env.step(actions,'INITIAL')
        else:
            env.step(actions,'NETWORK')
        log.logger.debug('time %d: reward = %.4f, steps = %d',
            time, episode_reward, episode)
        time+=time_step
        episode+=1
        actions.clear()  
        for agent in u_agents:
            ob_re[agent][1] = agent.get_reward()
        env.Update_Env(time,'NETWORK_LOAD')
    log.logger.info('\nfrom %d to %d by %d: average_reward = %.4f, episodes = %d',
        start_time, end_time, time_step, total_reward/episode, episode)
    env.close()
    return episode_reward

def play_episode_mode(env:Handover, u_agents, h_mode, mode=None,start_time=0, end_time=5000, time_step=50):
    total_reward = 0.0
    time = start_time
    episode = 0
    actions = {}
    env.reset(time,h_mode)
    ob_re = {}
    for agent in u_agents:
        ob_re[agent] = [agent.observe(h_mode),0.0]
    while(time<=end_time):
        episode_reward = 0.0
        for agent in u_agents:
            ob_re[agent][0] = agent.observe(h_mode)
            selected = [[i,ob_re[agent][0][i]] for i in range(agent.action_n) if ob_re[agent][0][i]>-1.0]
            max_num =-1.0
            action = -1
            for a in selected:
                if a[1]>max_num:
                    action = a[0]
                    max_num = a[1]
            actions[agent.user]= action+1
            episode_reward += ob_re[agent][1]
        total_reward += episode_reward
        if episode==0:
            env.step(actions,'INITIAL')
        else:
            env.step(actions,'NETWORK')
        log.logger.debug('time %d: reward = %.4f, steps = %d',
            time, episode_reward, episode)
        time+=time_step
        episode+=1
        actions.clear()  
        for agent in u_agents:
            ob_re[agent][1] = agent.get_reward()
        env.Update_Env(time,h_mode)
    log.logger.warning('time %d: average_reward = %.4f, steps = %d',
        time, total_reward/episode, episode)
    env.close()
    return episode_reward
