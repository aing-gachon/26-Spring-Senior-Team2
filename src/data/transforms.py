import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    EnsureTyped,
    RepeatChanneld
)

def get_transforms(image_size: int = 224):
    """
    강화학습 에이전트 전처리를 위한 MONAI Transform 파이프라인을 구성하여 반환합니다.
    Agent의 State로 입력되기 직전, Backbone(CNN)이 인식할 수 있는 고정 텐서로 변환합니다.
    """
    return Compose([
        # 1. 딕셔너리 키 "image"에 해당하는 경로에서 이미지를 로드합니다.
        LoadImaged(keys=["image"]),
        
        # 2. 이미지 차원 맨 앞에 채널 차원을 배치합니다 (1, H, W).
        EnsureChannelFirstd(keys=["image"]),
        
        # 3. ResNet50(Pretrained)은 컬러 사진(RGB, 3채널)을 요구하므로 흑백 1채널을 3개로 복사합니다.
        RepeatChanneld(keys=["image"], repeats=3),
        
        # 4. 픽셀 값을 스케일링하여 0~1 혹은 특정 범위로 정규화합니다.
        ScaleIntensityd(keys=["image"]),
        
        # 4. 신경망 입력 사이즈에 맞게 리사이징 (기본 224x224).
        Resized(keys=["image"], spatial_size=(image_size, image_size)),
        
        # 5. 파이토치 Tensor 형식으로 최종 변환합니다.
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ])
