import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.utils.utils import load_config, setup_logger
from src.data.dataset import get_dataloader
from src.data.transforms import get_transforms
from src.models.networks import FullDRLNetwork

def main():
    config = load_config("configs/config.yaml")
    logger = setup_logger("EvalLogger", os.path.join(config['paths']['log_dir'], "eval.log"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 평가 시에는 데이터 섞기(Shuffle) 없이 그대로 넣습니다
    transforms = get_transforms(config['env']['image_size'])
    dataloader = get_dataloader(
        img_dir=config['paths']['raw_image_dir'],
        transforms=transforms,
        target_disease=config['env']['target_disease'],
        normal_disease=config['env']['normal_disease'],
        target_count=config['env']['target_disease_count'],
        normal_ratio=config['env']['normal_ratio'],
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False 
    )
    
    # 빈 뇌(모델) 선언
    model = FullDRLNetwork(
        backbone_name=config['model']['backbone'],
        num_actions=config['model']['num_classes']
    )
    
    # 최고 기록을 달성한 가중치(Brain) 장착
    ckpt_path = os.path.join(config['paths']['checkpoint_dir'], "best_model.pth")
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval() # 추론 모드 전환 (Dropout이나 BatchNorm 영향 차단)
    
    y_true = []
    y_pred = []
    
    logger.info("Starting Evaluation Pipeline...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating (F1, G-Mean)"):
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Q-Value를 추론 후 가장 값이 높은 클래스를 최종 정답으로 지목
            q_values = model(images)
            preds = torch.argmax(q_values, dim=1).cpu().numpy()
            
            y_true.extend(labels)
            y_pred.extend(preds)
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. 1차 평가 : 전체 정확도 확인
    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Target Evaluation Accuracy: {acc:.4f}")
    
    # 2. 2차 평가 (의료 불균형 지표) : G-Mean 및 각 질병당 F1-score 확인
    report = classification_report(y_true, y_pred, zero_division=0)
    logger.info(f"Classification Report Details:\n{report}")
    
    # G-Mean 연산 로직 (기하 평균)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        g_mean = np.sqrt(sensitivity * specificity)
        logger.info(f"🎯 Imbalance Core Metric -> G-Mean: {g_mean:.4f}")
    else:
        logger.warning(f"🎯 Confusion Matrix varies. Unique classes: {np.unique(y_pred)}")
        
    print("\n✅ Evaluation Matrix Computed! Please check 'logs/eval.log'.")

if __name__ == "__main__":
    main()
