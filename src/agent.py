import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from src.networks import QNetwork
from src.replay_buffer import ReplayBuffer

class DDQNAgent:
    """
    고정된 타임스텝마다 Target Network를 업데이트하는 
    Double DQN 방식의 강화학습 에이전트.
    """
    def __init__(self, action_size=9, in_channels=1, batch_size=32, device="cpu", config=None):
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        
        # 모델 생성
        self.main_net = QNetwork(in_channels=in_channels, action_size=action_size).to(device)
        self.target_net = QNetwork(in_channels=in_channels, action_size=action_size).to(device)
        self.update_target_network()
        
        # 옵티마이저 등 설정
        lr = config['rl']['lr'] if config else 1e-4
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = config['rl']['gamma'] if config else 0.99
        self.epsilon = config['rl']['epsilon_start'] if config else 1.0
        self.epsilon_min = config['rl']['epsilon_end'] if config else 0.01
        self.epsilon_decay = config['rl']['epsilon_decay'] if config else 0.995
        
        buffer_size = config['rl']['buffer_size'] if config else 10000
        self.memory = ReplayBuffer(buffer_size)

    def update_target_network(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    def select_action(self, state):
        # Epsilon-Greedy Exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_net(state_tensor)
        return torch.argmax(q_values).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q(s, a)
        q_values = self.main_net(states).gather(1, actions)
        
        # Q'(s', a') with Target Network
        with torch.no_grad():
            next_actions = self.main_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon 감쇠
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
