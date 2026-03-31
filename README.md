# BicDRL-Reproduction: Imbalanced Medical Image Classification

본 프로젝트는 심층 강화학습(DRL) 에이전트를 활용하여 데이터 클래스 불균형이 극심한 `NIH Chest X-ray 14` 등 의료 영상 데이터를 효과적으로 평가 및 분류하는 파이프라인입니다. PyTorch와 MONAI 백본을 기반으로 개발되었습니다.

## 🚀 시작하기

### 파이프라인 구성 요소
- **Data (데이터셋)**: `/data/raw/` 경로에 `.png` 형식 이미지 배치. `Data_Entry_2017.csv` 파일은 `/data/` 경로에 배치.
- **Config**: 상세 하이퍼파라미터 및 경로는 `configs/config.yaml` 에서 관리합니다.
- **Backbone**: MONAI가 아닌 PyTorch `torchvision.models.resnet50` 과 맞춤화된 단일 채널(Gray-scale) 텐서를 지원합니다.

### 설치 방법
```bash
pip install -r requirements.txt
```

### 훈련 방식 (Training Run)
사전 설정된 파라미터를 기반으로 TD Loss 및 Epsilon Decay를 사용하는 DDQN 타깃 학습이 시작됩니다.
```bash
python train.py
```
