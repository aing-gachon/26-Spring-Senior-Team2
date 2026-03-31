import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import torch
from src.utils import calculate_iou, cxcywh_to_xyxy

class CXREnv(gym.Env):
    """
    DDQN 에이전트를 위한 Discrete Action CXR Bounding Box 탐색 환경.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, dataset, max_steps=10, patch_res=(224, 224)):
        super(CXREnv, self).__init__()
        self.dataset = dataset
        self.max_steps = max_steps
        self.patch_res = patch_res
        
        # Action Space: 9개의 Discrete Action
        # 0: Up, 1: Down, 2: Left, 3: Right
        # 4: Zoom In, 5: Zoom Out, 6: Aspect Ratio+, 7: Aspect Ratio-
        # 8: Terminate (탐색 확정)
        self.action_space = spaces.Discrete(9)
        
        # Observation Box (예: 단순화하여 현재 Crop 패치만 전달)
        # Global view와 결합하려면 Channels 차원을 확장하여 사용
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, *patch_res), dtype=np.uint8)

        self.current_step = 0
        self.image = None
        self.bbox_gt = None 
        self.current_window = [0.5, 0.5, 1.0, 1.0] # cx, cy, w, h
        self.current_index = 0
        
    def _get_obs(self):
        # Local patch 특징 추출
        ih, iw = self.image.shape
        cx, cy, w, h = self.current_window
        
        x_min = int(max(0, (cx - w/2) * iw))
        x_max = int(min(iw, (cx + w/2) * iw))
        y_min = int(max(0, (cy - h/2) * ih))
        y_max = int(min(ih, (cy + h/2) * ih))
        
        if x_max <= x_min or y_max <= y_min:
            crop = np.zeros(self.patch_res, dtype=np.uint8)
        else:
            crop = self.image[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                crop = np.zeros(self.patch_res, dtype=np.uint8)
            else:
                crop = cv2.resize(crop, self.patch_res)
            
        local_patch = np.expand_dims(crop, axis=0) # (1, H, W)
        return local_patch

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 데이터셋에서 샘플 로드
        self.current_index = np.random.randint(0, len(self.dataset))
        sample = self.dataset[self.current_index]
        
        self.image = sample["image"].squeeze() # (H, W)
        self.bbox_gt = sample["bbox"] # [x_min, y_min, x_max, y_max] / 1024.0 등 스케일 필요 시 정규화
        
        # 정답 BBox 정규화 (보상 계산용)
        # 이미지 가로/세로가 실제 1024인 경우라 가정 시
        h, w = self.image.shape
        if np.max(self.bbox_gt) > 1.0:
             self.bbox_gt = [self.bbox_gt[0]/w, self.bbox_gt[1]/h, (self.bbox_gt[0]+self.bbox_gt[2])/w, (self.bbox_gt[1]+self.bbox_gt[3])/h]
        else:
            # 원본 코드가 [x, y, w, h] 포맷이었으므로 변환
            dx, dy, dw, dh = self.bbox_gt
            self.bbox_gt = [dx, dy, dx+dw, dy+dh]

        self.current_window = [0.5, 0.5, 1.0, 1.0] 
        self.current_step = 0
        self.last_iou = calculate_iou(self.bbox_gt, cxcywh_to_xyxy(self.current_window))
        
        return self._get_obs(), {}

    def step(self, action):
        cx, cy, w, h = self.current_window
        alpha = 0.1 # 이동 보폭
        beta = 0.1 # 크기 조절 보폭
        
        terminated = False
        
        if action == 0: cy -= alpha
        elif action == 1: cy += alpha
        elif action == 2: cx -= alpha
        elif action == 3: cx += alpha
        elif action == 4: w -= beta; h -= beta
        elif action == 5: w += beta; h += beta
        elif action == 6: w += beta; h -= beta
        elif action == 7: w -= beta; h += beta
        elif action == 8: terminated = True
            
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        w = np.clip(w, 0.1, 1.0)
        h = np.clip(h, 0.1, 1.0)
        
        self.current_window = [cx, cy, w, h]
        self.current_step += 1
        
        truncated = False
        reward = 0.0
        
        curr_xyxy = cxcywh_to_xyxy(self.current_window)
        current_iou = calculate_iou(self.bbox_gt, curr_xyxy)
        iou_delta = current_iou - self.last_iou
        self.last_iou = current_iou
        
        # Reward Shaping
        reward += (np.sign(iou_delta) * 1.0) # IoU가 늘면 +1, 줄면 -1 방식의 단순 보상으로 DDQN 안정성 증대
        reward -= 0.1  # Step penalty
        
        if terminated:
            if current_iou > 0.5:
                reward += 5.0
            else:
                reward -= 5.0
                
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}
