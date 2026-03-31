import monai.transforms as mt

def get_transforms(image_size=224, phase='train'):
    """
    MONAI transforms 파이프라인 (Phase 2)
    """
    if phase == 'train':
        return mt.Compose([
            mt.LoadImaged(keys=["image"], image_only=True),
            mt.EnsureChannelFirstd(keys=["image"]),
            mt.Resized(keys=["image"], spatial_size=(image_size, image_size)),
            mt.ScaleIntensityd(keys=["image"]),
            mt.EnsureTyped(keys=["image"], dtype='float32'),
        ])
    else:
        return mt.Compose([
            mt.LoadImaged(keys=["image"], image_only=True),
            mt.EnsureChannelFirstd(keys=["image"]),
            mt.Resized(keys=["image"], spatial_size=(image_size, image_size)),
            mt.ScaleIntensityd(keys=["image"]),
            mt.EnsureTyped(keys=["image"], dtype='float32'),
        ])
