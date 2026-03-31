import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VNet2D(nn.Module):
    """
    의료영상 분석용 VNet의 2D 버전 (분류용 커스텀 모델)
    - 파라미터 수: 약 1.3M (경량)
    - 특징: Residual connections, Deep supervision(옵션) 및 의료영상 최적화
    """
    def __init__(self, num_classes, in_channels=1):
        super(VNet2D, self).__init__()
        
        self.in_conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Encoder Blocks (Residual)
        self.layer1 = self._make_layer(16, 32, stride=2)   # 112x112
        self.layer2 = self._make_layer(32, 64, stride=2)   # 56x56
        self.layer3 = self._make_layer(64, 128, stride=2)  # 28x28
        self.layer4 = self._make_layer(128, 256, stride=2) # 14x14
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_planes, out_planes, stride):
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_planes))
        
        shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        
        return nn.ModuleDict({
            'body': nn.Sequential(*layers),
            'shortcut': shortcut
        })

    def forward(self, x):
        x = self.bn1(F.relu(self.in_conv(x)))
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            out = layer['body'](x)
            out += layer['shortcut'](x)
            x = F.relu(out)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class DDQNNetwork(nn.Module):
    def __init__(self, num_classes, backbone_name="vnet"):
        super(DDQNNetwork, self).__init__()
        
        # Phase 3: CNN 백본 기반 특징 추출
        if backbone_name == "resnet50":
            # torchvision 제공 resnet50 활용
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # X-ray 이미지 1채널 처리
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif backbone_name == "vnet":
            # 의료영상용 VNet 2D 변형 모델 (경량)
            self.backbone = VNet2D(num_classes, in_channels=1)
            
        else:
            raise NotImplementedError(f"Backbone {backbone_name} is not implemented.")

    def forward(self, x):
        """
        Input: 상태 (State) 이미지 텐서 [B, C, H, W]
        Output: 행동 공간에 대한 Q-value 벡터 [B, num_classes]
        """
        return self.backbone(x)
