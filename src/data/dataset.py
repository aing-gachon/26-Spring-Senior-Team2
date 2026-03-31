import os
import random
from torch.utils.data import DataLoader
from monai.data import Dataset as MonaiDataset

def load_from_folders(img_dir: str, target_disease: str, normal_disease: str, target_count: int, normal_ratio: int):
    """
    수많은 데이터가 적힌 복잡한 CSV 파일을 전혀 쓰지 않고, 
    오직 사용자가 직관적으로 나눈 "폴더의 이름(Nodule, No Finding)" 자체가 정답(Label)이 되는 명쾌한 방식입니다.
    """
    target_dir = os.path.join(img_dir, target_disease)
    normal_dir = os.path.join(img_dir, normal_disease)
    
    # 1. 각 폴더 안의 이미지 가져오기
    target_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.png')] if os.path.exists(target_dir) else []
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.png')] if os.path.exists(normal_dir) else []
    
    # 2. 비율에 따른 추출 목표 개수 연산
    cnt_target = min(target_count, len(target_files))
    cnt_normal = min(cnt_target * normal_ratio, len(normal_files))
    
    if cnt_target == 0:
        print(f"🚨 Warning: '{target_dir}' 폴더에 이미지가 전혀 없습니다! 훈련을 시작할 수 없습니다.")
    
    # 3. 데이터 셔플링 후 추출
    # 고정 시드(42)를 주면, 매번 컴퓨터를 재부팅해도 똑같은 환자 X-ray 이미지만 공부하는 문제가 생기므로 시드를 뺐습니다.
    # 대신 dataloader에서 자체적으로 shuffle을 해줍니다.
    sampled_target = random.sample(target_files, cnt_target)
    sampled_normal = random.sample(normal_files, cnt_normal)
    
    # 4. 라벨 매핑 (0: 정상, 1: 질병)
    dynamic_labels = [normal_disease, target_disease]
    label_to_idx = {label: idx for idx, label in enumerate(dynamic_labels)}
    
    data_dicts = []
    for img_path in sampled_target:
        data_dicts.append({"image": img_path, "label": label_to_idx[target_disease]})
        
    for img_path in sampled_normal:
        data_dicts.append({"image": img_path, "label": label_to_idx[normal_disease]})
            
    print(f"✅ 폴더 기반 데이터 로드 성공: [{target_disease} 폴더]에서 {cnt_target}장 vs [{normal_disease} 폴더]에서 {cnt_normal}장 추출 완료.")
    return data_dicts, label_to_idx

def get_dataloader(img_dir: str, transforms, target_disease: str, normal_disease: str, target_count: int, normal_ratio: int, batch_size: int = 32, shuffle: bool = True):
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"원본 이미지 폴더를 찾을 수 없습니다: {img_dir}")
        
    data_dicts, _ = load_from_folders(img_dir, target_disease, normal_disease, target_count, normal_ratio)
    
    # MONAI 데이터셋 텐서화
    dataset = MonaiDataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4, 
        pin_memory=True
    )
    return dataloader
