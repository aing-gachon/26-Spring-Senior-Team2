import monai.transforms as mt
import torch

def get_base_transforms():
    """
    모든 데이터 파이프라인(Train, Eval)에 적용할 기본 MONAI 전처리 파이프라인.
    """
    transforms = mt.Compose([
        mt.ScaleIntensityd(keys=["image"]), # 0~1 정규화 
        mt.ToTensord(keys=["image", "label", "bbox"], dtype=torch.float32)
    ])
    return transforms

def get_train_transforms():
    """
    학습 시 추가로 사용할 Data Augmentation (Flip, Rotate 등)이 포함된 파이프라인
    """
    transforms = mt.Compose([
        mt.ScaleIntensityd(keys=["image"]),
        mt.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        mt.ToTensord(keys=["image", "label", "bbox"], dtype=torch.float32)
    ])
    return transforms
