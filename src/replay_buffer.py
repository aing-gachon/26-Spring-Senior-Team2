import random
import numpy as np

class ReplayBuffer:
    """
    DDQN의 Prioritized Experience Replay (PER) 또는 
    기본 Replay Buffer 구조를 위한 템플릿. 현재는 기본 버퍼 형태로 제공.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # 상태가 텐서 등일 경우 메모리 최적화를 위해 Numpy 변환 후 저장 필요
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
