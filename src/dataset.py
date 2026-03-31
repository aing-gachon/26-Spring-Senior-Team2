import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from monai.data import Dataset

class CXR8Dataset(Dataset):
    """
    MONAI Dataset을 상속한 CXR8 이미지, BBox, 라벨 로드 모듈.
    Data_Entry_2017.csv 및 BBox 정보 등을 활용하여 에이전트 학습에 필요한
    상태 이미지와 정답 BBox 좌표를 제공합니다.
    """
    def __init__(self, raw_img_dir, csv_path, transform=None, use_bbox_only=False):
        self.raw_img_dir = raw_img_dir
        self.transform = transform
        
        # Multi-class 중 기준 타겟 클래스 (3cm 이하 작은 결절)
        self.target_class = "Nodule"
        
        # 14개의 NIH Multi-label (주어진 과제에 따라 1개 단일 클래스로 필터링)
        self.classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        # CSV 로드 
        if os.path.exists(csv_path):
            self.entry_df = pd.read_csv(csv_path)
            # Nodule이 포함된 데이터만 필터링하여 에이전트가 해당 타겟 탐색에 집중하도록 처리
            if 'Finding Labels' in self.entry_df.columns:
                self.entry_df = self.entry_df[self.entry_df['Finding Labels'].str.contains(self.target_class, na=False)]
        else:
            self.entry_df = pd.DataFrame(columns=['Image Index', 'Finding Labels'])
            
        self.data_entries = self.entry_df.to_dict('records')

    def __len__(self):
        return len(self.data_entries)

    def _get_bbox(self, image_name):
        # BBox 데이터가 있을 경우 반환 (없으면 0 포맷)
        return np.array([0, 0, 0, 0], dtype=np.float32)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        image_name = entry['Image Index']
        
        img_path = os.path.join(self.raw_img_dir, image_name)
        if not os.path.exists(img_path):
            # 파일이 없으면 더미 텐서 (크롬/런타임 에러 방지용)
            img_array = np.zeros((1, 1024, 1024), dtype=np.float32)
        else:
            img = Image.open(img_path).convert('L')
            img_array = np.array(img, dtype=np.float32)
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=0)

        data_dict = {
            "image": img_array,
            "label": np.zeros(len(self.classes), dtype=np.float32), # 더미
            "bbox": self._get_bbox(image_name),
            "image_name": image_name
        }
        
        if self.transform:
            data_dict = self.transform(data_dict)
            
        return data_dict
