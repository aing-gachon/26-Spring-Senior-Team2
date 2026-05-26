import monai.transforms as mt

def get_transforms(image_size=224, phase='train'):
    """
    MONAI transforms 파이프라인 (Phase 2)
    """
    base_transforms = [
        mt.LoadImaged(keys=["image"], image_only=True),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Lambdad(keys=["image"], func=lambda x: x[0:1, ...]), # RGBA/RGB 등 다중 채널 방지 (1채널 고정)
        mt.Resized(keys=["image"], spatial_size=(image_size, image_size)),
        mt.ScaleIntensityd(keys=["image"]),
    ]

    if phase == 'train':
        # 데이터 증강 추가
        base_transforms.extend([
            mt.RandRotated(keys=["image"], range_x=0.15, prob=0.5, keep_size=True),
            mt.RandFlipd(keys=["image"], spatial_axis=[0, 1], prob=0.5),
            mt.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.3)),
            mt.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        ])

    base_transforms.append(mt.EnsureTyped(keys=["image"], dtype='float32'))
    
    return mt.Compose(base_transforms)