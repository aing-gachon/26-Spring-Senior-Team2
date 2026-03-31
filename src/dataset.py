import os
import math
import numpy as np
import pandas as pd
from monai.data import Dataset, DataLoader
from .transforms import get_transforms

def extract_label(finding):
    # Dummy binary extraction for setup purpose (0: Normal, 1: Finding)
    if pd.isna(finding) or 'No Finding' in finding:
        return 0
    return 1

def load_and_filter_data(csv_path, image_dir, epsilon=1e-5):
    """
    데이터셋 필터링 밑 적응형 가중치(gamma_k) 사전 계산 (Phase 1)
    """
    data_dicts = []
    class_counts = {0: 0, 1: 0} # Binary example
    
    if not os.path.exists(csv_path):
        print(f"[Warning] CSV not found at {csv_path}. Using dummy dictionary for setup testing.")
        # 더미 이미지 생성 (MONAI 로딩 테스트용)
        if not os.path.exists("dummy.png"):
            from PIL import Image
            img = Image.new('L', (256, 256), color=128)
            img.save("dummy.png")
            
        # Dummy data for pipeline validation
        class_counts = {0: 100, 1: 10} # Imbalanced scenario
        for i in range(110):
            lbl = 0 if i < 100 else 1
            data_dicts.append({"image": "dummy.png", "label": lbl})
    else:
        df = pd.read_csv(csv_path)
        # Simplified: Filter single finding
        df['label'] = df['Finding Labels'].apply(extract_label)
        
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row['Image Index'])
            lbl = row['label']
            data_dicts.append({"image": img_path, "label": lbl})
            class_counts[lbl] += 1
            
    # Calculate adaptive weights (gamma_k = 1 / ln(c_k + epsilon))
    class_weights = {}
    for cls_idx, count in class_counts.items():
        if count > 0:
            class_weights[cls_idx] = 1.0 / math.log(count + epsilon)
        else:
            class_weights[cls_idx] = 1.0
            
    print(f"Data Loaded: {len(data_dicts)} files. Class counts: {class_counts}. Weights: {class_weights}")
    return data_dicts, class_weights

def get_dataloader(data_dicts, batch_size=32, image_size=224, phase='train'):
    transforms = get_transforms(image_size, phase=phase)
    dataset = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(phase=='train'), num_workers=4)
    return loader
