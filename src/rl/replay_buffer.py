import numpy as np
import torch

class PrioritizedReplayBuffer:
    """
    PER(Prioritized Experience Replay) 버퍼.
    예측과 정답의 차이(TD-Error)가 큰 인스턴스를 더 높은 가중치로 자주 학습하도록 강제합니다.
    극심한 의료 데이터 불균형 환경에서 오답률을 낮추는 핵심 로직입니다.
    """
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        # 우선순위가 샘플링에 미치는 영향력을 조절하는 지수 (alpha)
        self.alpha = alpha
        
        self.buffer = []
        # 저장된 샘플들의 에러 중요도(Priority) 수치를 기록하는 배열
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """
        새로운 데이터를 큐(Queue)에 추가합니다. 
        처음 들어온 데이터는 무조건 한 번은 뽑혀서 평가받아야 하므로 "최대 우선순위"를 부여합니다.
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        # 버퍼가 꽉 차지 않았을 때
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        # 꽉 찼으면 과거 메모리 덮어쓰기 로직
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        # 포인터 위치에 최대 우선순위 기록
        self.priorities[self.pos] = max_priority
        # 원형 큐 포인터 이동 로직
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=32, beta=0.4):
        """
        우선순위(확률)에 맞춰서 편향적으로 데이터를 골라 미니배치(Mini Batch)를 리턴합니다.
        가중치 보정을 위한 Importance Sampling (IS) Weights 도 함께 제공합니다.
        """
        if len(self.buffer) == 0:
            return [], [], []
            
        # 현재까지 차있는 버퍼만큼만 계산
        prios = self.priorities[:len(self.buffer)]
        
        # 확률(P) = p^a / sum(p^a)
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # 위 확률 분포를 바탕으로 랜덤 추출
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # IS Weights(편향을 수식적으로 억제하는 가중치) 계산: w = (N * P)^-beta
        total_len = len(self.buffer)
        weights = (total_len * probs[indices]) ** (-beta)
        weights /= weights.max()  # 스케일 정규화
        
        # 언패킹 및 PyTorch 텐서 포맷화
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # states 요소들은 shape이 (1, C, H, W) 이므로 합치면 (Batch, C, H, W) 가 됨
        states = torch.cat(states, dim=0)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return (states, actions, rewards, next_states, dones), indices, torch.FloatTensor(weights)

    def update_priorities(self, batch_indices, batch_priorities):
        """
        에이전트가 한 번 학습하고 뱉은 새로운 오답률(TD-Error) 수치로 버퍼에 저장된 과거 중요도를 갱신.
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            # 오차 보정용 아주 작은 엡실론 값을 추가하여 무한 0이 되는 것을 방지
            self.priorities[idx] = float(priority) + 1e-5
