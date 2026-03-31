import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MedicalImageEnv(gym.Env):
    """
    의료 이미지 분류를 위한 강화학습 커스텀 환경(Environment)입니다.
    데이터로더 내의 전처리된 배치 이미지를 순차적으로 1개씩 에이전트에게 상태(State)로 전달하고,
    에이전트의 액션 판별값과 정답 라벨을 비교하여 보상(Reward)을 리턴합니다.
    """
    def __init__(self, dataloader, num_classes=2, max_steps_per_episode=1000):
        super(MedicalImageEnv, self).__init__()
        # DataLoader Iterator 매핑
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
        
        # Action Space: 동적 할당 (0: 정상, 1: 타겟 질병)
        self.action_space = spaces.Discrete(num_classes)
        
        # State Space: MONAI 전처리를 거친 (3, 224, 224) 텐서 (명시적 규격용)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 224, 224), dtype=np.float32)
        
        self.max_steps = max_steps_per_episode
        self.current_step = 0
        self.current_batch = None
        self.current_idx = 0

    def _get_next_data(self):
        """배치에서 이미지와 라벨을 1건씩 추출해 리턴합니다. (Batch 소진 시 다음 Batch 로드)"""
        if self.current_batch is None or self.current_idx >= len(self.current_batch['image']):
            try:
                self.current_batch = next(self.data_iter)
            except StopIteration:
                # 데이터 1회를 다 돌았다면(1 Epoch 소진), 다시 초기화
                self.data_iter = iter(self.dataloader)
                self.current_batch = next(self.data_iter)
            self.current_idx = 0
            
        # 단일 이미지 추출 및 차원 확장 (에이전트가 Batch=1 형태로 받기 위함)
        image = self.current_batch['image'][self.current_idx].unsqueeze(0)
        label = self.current_batch['label'][self.current_idx].item()
        
        self.current_idx += 1
        return image, label

    def reset(self, seed=None, options=None):
        """에피소드를 초기화하고 첫 번째 State 텐서를 반환합니다."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # 옵션으로 iter 리셋 등을 처리할 수 있으나 기본적으로 계속 진행
        self.current_image, self.current_label = self._get_next_data()
        
        info = {}
        return self.current_image, info

    def step(self, action):
        """
        에이전트가 단일 이미지에 대해 제시한 행동(Action, 특정 질병 지목)에 대한 결과를 처리합니다.
        """
        self.current_step += 1
        
        # 논문 구현: 차등 보상(Adaptive Reward) 체계
        # 정답(Normal) 시 기본 +1, 소수 희귀 질환(Nodule, 1) 적중 시 압도적 보상(+50)
        if action == self.current_label:
            if self.current_label == 1:
                reward = 50.0
            else:
                reward = 1.0
        else:
            reward = -1.0
            
        # 에피소드 종료 지점(Truncation, Max Steps 도달 시) 여부 판정
        terminated = False
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            
        # 다음 스텝을 위해 새로운 정보 (Next State) 로드
        next_image, next_label = self._get_next_data()
        
        # 현재 클래스에 새로 로드된 정보 덮어씌움
        self.current_image = next_image
        self.current_label = next_label
        
        # 디버깅/로깅에 필요한 부가 정보
        info = {'real_label': self.current_label}
        
        return self.current_image, reward, terminated, truncated, info
