import os
import math
import numpy as np
import pandas as pd
from monai.data import Dataset, DataLoader
from .transforms import get_transforms

def extract_label(finding):
    # Nodule 이진 분류 ('No Finding' -> 0, 'Nodule' 단일 병변 -> 1, 그 외 복합/타질병 -> -1 무시)
    if pd.isna(finding) or finding == 'No Finding':
        return 0
    elif finding == 'Nodule':
        return 1
    return -1

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
        # Nodule이나 No Finding 단일 병변만 남기도록 필터링
        df['label'] = df['Finding Labels'].apply(extract_label)
        
        # 드라이브에서 파일 접근 속도를 높이기 위해 미리 폴더 내 파일명 집합을 만듭니다.
        existing_files = {
            0: set(os.listdir(os.path.join(image_dir, "No Finding"))) if os.path.exists(os.path.join(image_dir, "No Finding")) else set(),
            1: set(os.listdir(os.path.join(image_dir, "Nodule"))) if os.path.exists(os.path.join(image_dir, "Nodule")) else set()
        }

        for _, row in df.iterrows():
            lbl = row['label']
            if lbl == -1:
                continue # 설정한 타겟 질환이 아닌 경우스킵
            
            img_filename = row['Image Index']
            
            # 실제 파일이 드라이브(또는 폴더)에 존재하는지 검사 (일부 데이터만 올렸을 경우 대비)
            if img_filename not in existing_files[lbl]:
                continue
            
            # data/raw/Nodule 혹은 data/raw/No Finding 폴더 구조 반영
            folder_name = "Nodule" if lbl == 1 else "No Finding"
            img_path = os.path.join(image_dir, folder_name, img_filename)
            
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(phase=='train'), num_workers=0)
    return loader
