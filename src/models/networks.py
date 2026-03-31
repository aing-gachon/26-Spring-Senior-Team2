import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    """
    사전 학습된 ResNet을 이용해 원본 의료 이미지로부터 특징을 압축하여 추출합니다.
    CNN 연산을 통해 추출된 Feature Map (HeatMap) 속성을 강화학습 에이전트가 이해할 수 있는
    상태(State Vector)로 요약합니다.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(FeatureExtractor, self).__init__()
        
        # 권장: torchvision >= 0.13 의 weights 매개변수를 쓰나 호환성을 위해 pretrained 파라미터 유지
        if backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = resnet.fc.in_features  # 2048
        elif backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_dim = resnet.fc.in_features  # 512
        else:
            raise ValueError("Supported backbones: resnet18, resnet50")
            
        # 마지막 분류(Classification FC) 계층 전까지만 떼어내어 사용
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """
        x: 전처리된 이미지 텐서 (Batch_size, Channel, H, W)
        반환값: 추출 및 압축된 특징 벡터 (Batch_size, feature_dim)
        """
        features_map = self.backbone(x)  # (Batch_size, feature_dim, 1, 1)
        state_vector = torch.flatten(features_map, 1)  # (Batch_size, feature_dim)
        return state_vector


class QNetwork(nn.Module):
    """
    강화학습 DDQN 에이전트의 중심망 (MainNet / TargetNet 역할)
    추출된 특징 벡터를 입력(State)으로 받아 각 행동(어떤 질병인지 Class 선택)의 Q-Value를 모구 계산합니다.
    """
    def __init__(self, feature_dim, num_actions):
        super(QNetwork, self).__init__()
        
        # 간단한 다층 퍼셉트론(MLP) 기반 헤드 구성
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, state_features):
        """
        반환값: 각 15개 클래스 판별에 대한 Q-Value 기대값 텐서 (Batch_size, num_actions)
        """
        return self.fc(state_features)


class FullDRLNetwork(nn.Module):
    """
    Feature Extractor(Backbone)와 Q-Network(Head)를 하나로 묶어놓은 완성본 모듈입니다.
    """
    def __init__(self, backbone_name='resnet50', num_actions=15):
        super(FullDRLNetwork, self).__init__()
        self.extractor = FeatureExtractor(backbone_name=backbone_name)
        self.q_head = QNetwork(feature_dim=self.extractor.feature_dim, num_actions=num_actions)
        
    def forward(self, x):
        # 1. 이미지에서 특징 추출 (상태 텐서 도출)
        features = self.extractor(x)
        
        # 2. 상태 텐서를 기반으로 각 행위 확률(Q-value) 계산
        q_values = self.q_head(features)
        
        return q_values
