import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """경험 튜플 (s_t, a_t, r_t, s_{t+1}) 버퍼에 저장"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """무작위 배치 크기만큼 샘플 추출 (추후 PER로 고도화 예정)"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 텐서 배치 변환
        state_batch = torch.cat([s for s in states])
        next_state_batch = torch.cat([s for s in next_states])
        action_batch = torch.tensor(actions)
        reward_batch = torch.tensor(rewards)
        done_batch = torch.tensor(dones)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        
    def __len__(self):
        return len(self.buffer)
