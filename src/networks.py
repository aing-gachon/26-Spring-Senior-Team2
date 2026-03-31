import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

class CXRClassifier(nn.Module):
    """
    Phase 0/3 용 정밀 진단 모델 (기존 classifier.py 내용 유지)
    """
    def __init__(self, spatial_dims=2, in_channels=1, out_classes=14):
        super(CXRClassifier, self).__init__()
        self.model = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_classes
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))

class QNetwork(nn.Module):
    """
    DDQN(Double DQN)을 위한 Q-Network.
    이미지 입력을 받아 행동(Action) 스페이스에 대한 각 Q-value를 출력합니다.
    """
    def __init__(self, in_channels=1, action_size=9): 
        # Action 예: Up, Down, Left, Right, Zoom_in, Zoom_out, Aspect_ratio_+, Aspect_ratio_-, Terminate
        super(QNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 25 * 25, 512), # 224x224 기준 대략적 차원 (환경에 맞게 조정 필요)
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
