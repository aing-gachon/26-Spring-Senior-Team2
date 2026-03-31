인성
insung0615
온라인

#2팀 채널의 시작이에요. 
유시현/소웨/20 — 2026-03-10 오후 7:21
https://github.com/Project-MONAI/tutorials/tree/main/detection
GitHub
tutorials/detection at main · Project-MONAI/tutorials
MONAI Tutorials. Contribute to Project-MONAI/tutorials development by creating an account on GitHub.
MONAI Tutorials. Contribute to Project-MONAI/tutorials development by creating an account on GitHub.
류건우/인지/21 [ZZZ],  — 2026-03-10 오후 7:34
../tree/main/detection
../blob/main/3d_segmentation/brats_segmentation_3d.ipynb
등 사용하여 종양 위치 찾기 할 예정
이후 RL 적용
[관련 데이터셋]
https://www.cancerimagingarchive.net/access-data/
https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest?hl=ko
The Cancer Imaging Archive (TCIA)
Jeff Tobler
Access the Data - The Cancer Imaging Archive (TCIA)
Access the Data - The Cancer Imaging Archive (TCIA)
Google Cloud Documentation
NIH 흉부 X선 데이터 세트  |  Cloud Healthcare API  |  G...
NIH 흉부 X선 데이터 세트  |  Cloud Healthcare API  |  G...
이미지
 [ZZZ], 
류건우/인지/21 [ZZZ],  — 2026-03-10 오후 7:42
3d_segmentation/torch 디렉토리에 있는 unet_training_array.py 실행 결과

예제 코드가 정상적으로 훈련 파이프라인(가상 3D 이미지 데이터 생성, 3D U-Net 신경망 모델, 평가 지표 계산 등)을 수행함을 확인함

훈련이 진행되며 Loss 값이 0.61에서 약 0.39까지 순조롭게 감소하였으며, 학습의 성능 지표인 Dice score(정확도) 역시 평가 기준을 넘어서 최고 0.9376까지 기록
유시현/소웨/20 — 2026-03-10 오후 7:48
종양위치 MONAI로 분류 후 종양 위치 Label data를 기반으로 HeatMap 작성 후 RL을 통해서 종양 위치 탐색(location이 아닌 Probablility로 계산 할듯)
종양위치 MONAI로 분류 후 종양 위치 Label data를 기반으로
메시지 12개 ›
스레드에 새 메시지가 없어요.
18일 전
송이두/컴공/22
 님이 종양위치 MONAI로 분류 후 종양 위치 Label data를 기반으로 스레드를 시작하셨어요(스레드 모두 보기). — 2026-03-12 오후 5:50
류건우/인지/21
 님이 New 주제 스레드를 시작하셨어요(스레드 모두 보기). — 2026-03-24 오후 7:12
New 주제
메시지 14개 ›
유시현/소웨/20
6일 전
﻿
류건우/인지/21 [ZZZ],  — 2026-03-24 오후 7:12
.
[상세 구현 계획]
## 실험 재현 파이프라인 (BicDRL 프레임워크 기반)
#### 1. 데이터셋 정제 및 단순화 (다중 클래스/이진 분류화)

NIH Chest X-ray 14 데이터셋(`Data_Entry_2017_v2020.csv`)의 다중 라벨(Multi-label) 특성을 에이전트가 학습할 수 있는 환경(MDP)에 맞게 다중 클래스(Multi-class) 또는 이진 분류(Binary) 형태로 단순화합니다.

- **Single-finding 필터링:** 한 이미지에 여러 질병이 있는 케이스는 제외하고, 오직 단일 질병(예: 'Pneumothorax'만 있는 경우) 혹은 정상(No Finding)인 데이터만 추출합니다. (논문에서도 Chest X-ray 데이터셋을 이진 분류 테스크로 구성하여 실험했습니다 ).
    
- **라벨 인코딩:** 추출된 질병명을 기반으로 클래스 인덱스(0, 1, 2 등)를 부여합니다. 이 값은 에이전트가 행동을 취한 후 정답 여부를 판별하기 위한 **실제 정답 라벨($b_t$)**로만 사용됩니다.
    
- **클래스 가중치 계산:** 학습 전, 각 클래스별 전체 샘플 수($c_k$)를 카운트하여 보상 함수에 쓰일 적응형 가중치 $\gamma_k = 1 / \ln(c_k + \epsilon)$를 미리 계산해 둡니다.
    

#### 2. 의료 이미지 전처리 (MONAI 활용 상태(State) 구성)

에이전트에게 환경의 상태($s_t$)로 넘겨줄 원본 이미지를 딥러닝 연산에 적합한 텐서로 변환합니다. (오해의 소지가 있었던 Dice Loss 등 분할/크롭 개념은 완전히 배제합니다.)

- `LoadImaged`: 원본 X-ray 이미지 파일을 로드합니다.
    
- `EnsureChannelFirstd`: 이미지의 채널 차원을 배치합니다 (예: `[1, H, W]`).
    
- `Resized`: 네트워크 입력 규격에 맞게 해상도를 고정합니다 (예: 224x224).
    
- `ScaleIntensityd`: 모델 학습 안정성을 위해 픽셀 값을 0~1 사이로 정규화합니다.
    
- `EnsureTyped`: 파이토치(PyTorch) Tensor 자료형으로 최종 변환합니다. 이 결과물이 에이전트가 관찰할 **상태($s_t$)**가 됩니다.
    

#### 3. 에이전트 아키텍처 설계 (CNN 결합형 Q-Network)

가장 중요한 구조적 변경입니다. 단순 평탄화(Flatten)가 아닌, 이미지 특징을 추출할 수 있도록 에이전트 내부의 Q-Network(`MainNet` 및 `TargetNet`)에 CNN 백본을 결합합니다.

- **특징 추출기 (Feature Extractor):** MONAI에서 제공하는 `ResNet50`이나 `DenseNet121`과 같은 CNN 모델을 가져와 `MainNet`의 앞단(Backbone)으로 배치합니다. (단, 마지막 분류 Layer는 제거합니다).
    
- **Q-Value 출력 계층 (FC Layer):** CNN을 통과해 나온 특징 벡터(Feature Vector)를 Fully Connected Layer에 연결합니다. 이 계층의 최종 출력 노드 수는 우리가 정의한 **클래스(행동 공간 $\mathcal{A}$)의 개수**와 동일해야 합니다.
    
- 최종적으로 네트워크는 상태 $s_t$를 입력받아 각 행동(클래스 예측)에 대한 기대 보상값인 Q-value를 출력합니다.
    

#### 4. 보상 함수(Reward Function) 구현

논문의 핵심인 DDQNbic 알고리즘의 보상 체계를 코드로 구현합니다.

- 에이전트가 예측한 라벨($a_t$)과 1단계에서 만든 정답 라벨($b_t$)을 비교합니다.
    
- 일치하면($a_t = b_t$), 긍정적인 보상 $r_t = \gamma_k$를 부여합니다.
    
- 불일치하면($a_t \neq b_t$), 부정적인 보상 $r_t = -\gamma_k$를 부여합니다.
    
- 소수 클래스일수록 $\gamma_k$ 값이 크므로, 소수 클래스를 맞췄을 때 더 큰 보상을, 틀렸을 때 더 큰 페널티를 받게 됩니다.
    

#### 5. DDQN 훈련 루프 실행

- 배치 단위로 $s_t$를 Q-Network에 입력하여 Epsilon-greedy 전략으로 행동 $a_t$를 선택합니다.
    
- 환경(보상 함수)으로부터 보상 $r_t$를 받고, 튜플 $(s_t, a_t, r_t, s_{t+1})$을 경험 재생 버퍼(Replay Buffer)에 저장합니다. 논문과 동일하게 Prioritized Experience Replay(PER)를 적용하여 TD 에러가 큰 샘플을 우선적으로 학습시킵니다.
    
- Adam 옵티마이저를 사용해 `MainNet`을 업데이트하고, 주기적으로 `TargetNet`에 가중치를 복사(Soft/Hard update)하여 과대평가(Overestimation)를 방지합니다.

pipeline.md
5KB
[단순화 (Readme)]
# 🚀 BicDRL-Reproduction: Deep Reinforcement Learning for Imbalanced Medical Image Classification

## 📌 Project Overview

본 프로젝트는 심층 강화학습(DRL)을 활용하여 의료 이미지 데이터의 극심한 클래스 불균형(Class Imbalance) 문제를 해결하는 **BicDRL 프레임워크**를 PyTorch와 MONAI를 기반으로 구현하고 재현하는 것을 목표로 합니다.

에이전트(Agent)가 소수 클래스(희귀 병변)를 정확히 분류했을 때 더 높은 보상을 주는 적응형 보상 메커니즘을 적용하며, **NIH Chest X-ray 14** 데이터셋을 활용하여 학습 파이프라인을 구축합니다.

---

## 🏗️ System Pipeline

### Phase 1. Data Preparation & Simplification (데이터 정제 및 환경 구축)

NIH 데이터셋의 다중 라벨(Multi-label) 특성을 강화학습 에이전트의 단일 행동(Action) 공간에 맞게 단순화합니다.

- **단일 병변 필터링:** `Data_Entry_2017_v2020.csv`에서 단일 질병(예: Pneumothorax)만 가진 케이스와 정상(No Finding) 케이스를 추출하여 이진 분류(Binary Classification) 또는 다중 클래스(Multi-class) 데이터셋으로 구성합니다.
    
- **라벨 인코딩 (Metadata):** 에이전트의 보상 평가를 위한 실제 정답 라벨($b_t$)로 사용할 수 있도록 타깃 질병을 인코딩합니다.
    
- **적응형 클래스 가중치 계산:** 학습 전 전체 훈련 데이터의 클래스 분포를 분석하여, 보상 함수에 사용할 적응형 가중치 $\gamma_k = 1 / \ln(c_k + \epsilon)$ 를 사전 계산합니다. ($c_k$: 해당 클래스의 샘플 수)
    

### Phase 2. Medical Image Preprocessing (MONAI 기반 전처리)

원시 의료 이미지(Raw Data)를 에이전트가 관찰할 수 있는 상태(State, $s_t$) 텐서로 정제합니다.

- `LoadImaged`: X-ray 이미지 원본을 로드합니다.
    
- `EnsureChannelFirstd`: 딥러닝 연산 규격에 맞춰 채널 차원을 배치합니다.
    
- `Resized`: 네트워크 입력 크기(예: `224x224`)에 맞춰 해상도를 조정합니다.
    
- `ScaleIntensityd`: 모델 학습의 안정성과 속도 향상을 위해 픽셀 값을 `0~1` 스케일로 정규화합니다.
    
- `EnsureTyped`: 최종적으로 PyTorch Tensor 형태로 변환하여 에이전트 환경(Environment)에 전달할 준비를 마칩니다.
    

### Phase 3. DRL Agent Architecture (CNN 기반 DDQN 설계)

의료 이미지의 공간적 특징을 추출하기 위해 Q-Network 내부에 CNN 백본을 결합한 에이전트를 설계합니다.

- **특징 추출기 (Feature Extractor):** `ResNet50` 또는 `DenseNet121` 구조를 차용하여 `MainNet`과 `TargetNet`의 전면부에 배치합니다. (단, 기존의 마지막 분류 계층은 제거합니다).
    
- **Q-Value 출력 계층 (FC Layer):** 추출된 특징 벡터(Feature Vector)를 입력받아, 최종적으로 에이전트의 행동 공간(정의된 클래스 개수) 크기와 동일한 개수의 Q-value를 출력하도록 선형 계층을 구성합니다.
    

### Phase 4. Environment & Reward Mechanism (환경 및 보상 체계)

에이전트의 예측과 실제 정답을 비교하여 클래스 불균형을 해소하는 핵심 강화학습 루프입니다.

- **상태 관찰 ($s_t$):** Phase 2에서 전처리된 이미지 텐서를 에이전트가 관찰합니다.
    
- **행동 선택 ($a_t$):** $\epsilon$-greedy 전략에 따라 이미지가 속할 클래스를 예측(Action)합니다.
    
- **보상 계산 ($r_t$):** * 정답 시 ($a_t = b_t$): $r_t = \gamma_k$ (소수 클래스일수록 높은 긍정적 보상)
    
    - 오답 시 ($a_t \neq b_t$): $r_t = -\gamma_k$ (소수 클래스일수록 높은 페널티)
        

### Phase 5. Training Loop (모델 학습 및 최적화)

DDQN(Double Deep Q-Network) 알고리즘을 기반으로 에이전트를 학습시킵니다.

- **Prioritized Experience Replay (PER):** 튜플 $(s_t, a_t, r_t, s_{t+1})$ 을 버퍼에 저장하고, TD 에러가 큰(학습이 더 필요한) 샘플을 우선적으로 추출하여 학습합니다.
    
- **네트워크 업데이트:** Adam 옵티마이저를 사용하여 `MainNet`을 업데이트하며, Q-value의 과대평가를 방지하기 위해 주기적으로 `TargetNet`에 가중치를 동기화합니다.

readme.md
5KB
[프레임워크 구조]
이미지
[파일 구조]
# 📂 Project Directory Structure

본 프로젝트는 PyTorch와 MONAI를 기반으로 심층 강화학습(DRL) 에이전트를 학습시키기 위해 모듈화된 파일 구조를 가집니다.

```text
BicDRL-Reproduction/
│
├── data/                       # 데이터셋 및 메타데이터 폴더 (gitignore 권장)
│   ├── raw/                    # 원본 의료 이미지 (e.g., NIH Chest X-ray images)
│   ├── processed/              # 전처리된 데이터 또는 분할된 데이터 (선택적)
│   └── Data_Entry_2017.csv     # 정답 라벨이 포함된 원본 메타데이터 CSV 파일
│
├── configs/                    # 하이퍼파라미터 및 설정 파일 폴더
│   └── config.yaml             # 학습률, 배치 사이즈, ε-decay, 경로 등 통합 설정
│
├── src/                        # 핵심 소스 코드 폴더
│   ├── __init__.py
│   ├── dataset.py              # 데이터 필터링(다중 라벨->단일 클래스) 및 MONAI DataLoader 구성
│   ├── transforms.py           # MONAI 전처리 파이프라인 (Resize, Normalize, EnsureType 등)
│   ├── networks.py             # CNN 백본(ResNet 등)과 결합된 Q-Network (MainNet, TargetNet) 구조
│   ├── agent.py                # DDQN 에이전트 클래스, 행동 선택(ε-greedy) 및 네트워크 업데이트 로직
│   ├── replay_buffer.py        # 경험 재생 버퍼 (Prioritized Experience Replay - PER 구현)
│   ├── environment.py          # 강화학습 환경 구축 (상태 관찰, 정답 비교, 적응형 보상 함수 구현)
│   └── utils.py                # 클래스 가중치(γ_k) 계산, 로깅, 시드(Seed) 고정 등 유틸리티 함수
│
├── train.py                    # 에이전트 학습을 실행하는 메인 스크립트 (Training Loop)
├── evaluate.py                 # 학습된 모델의 성능 평가 (F1-score, G-mean 계산 및 테스트)
│
├── checkpoints/                # 학습 중 저장되는 모델 가중치(.pth) 폴더
│   └── best_model.pth          # 가장 성능이 좋은 모델 파일
│
├── notebooks/                  # EDA 및 실험용 주피터 노트북 폴더
│   └── 01_data_exploration.ipynb # 데이터 분포 확인 및 MONAI 전처리 테스트용 노트북
│
├── requirements.txt            # 프로젝트 실행에 필요한 패키지 목록 (PyTorch, MONAI, Pandas 등)
└── README.md                   # 프로젝트 개요, 파이프라인 설명 및 실행 방법 안내

Directory Structure.md
3KB
+) Multi-class를 위한 기준 class
: Nodule - 3cm 이하 작은 결절
+) 깃허브 싹 다 바꿔야 됨
https://github.com/aing-gachon/26-Spring-Senior-Team2
유시현/소웨/20 — 2026-03-24 오후 7:21
첨부 파일 형식: acrobat
A_Deep_Reinforcement_Learning_Framework_for_Imbalanced_Medical_Image_Classification.pdf
5.75 MB
https://github.com/project-monai/monai monai official repo
GitHub
GitHub - Project-MONAI/MONAI: AI Toolkit for Healthcare Imaging
AI Toolkit for Healthcare Imaging. Contribute to Project-MONAI/MONAI development by creating an account on GitHub.
GitHub - Project-MONAI/MONAI: AI Toolkit for Healthcare Imaging
﻿
# 📂 Project Directory Structure

본 프로젝트는 PyTorch와 MONAI를 기반으로 심층 강화학습(DRL) 에이전트를 학습시키기 위해 모듈화된 파일 구조를 가집니다.

```text
BicDRL-Reproduction/
│
├── data/                       # 데이터셋 및 메타데이터 폴더 (gitignore 권장)
│   ├── raw/                    # 원본 의료 이미지 (e.g., NIH Chest X-ray images)
│   ├── processed/              # 전처리된 데이터 또는 분할된 데이터 (선택적)
│   └── Data_Entry_2017.csv     # 정답 라벨이 포함된 원본 메타데이터 CSV 파일
│
├── configs/                    # 하이퍼파라미터 및 설정 파일 폴더
│   └── config.yaml             # 학습률, 배치 사이즈, ε-decay, 경로 등 통합 설정
│
├── src/                        # 핵심 소스 코드 폴더
│   ├── __init__.py
│   ├── dataset.py              # 데이터 필터링(다중 라벨->단일 클래스) 및 MONAI DataLoader 구성
│   ├── transforms.py           # MONAI 전처리 파이프라인 (Resize, Normalize, EnsureType 등)
│   ├── networks.py             # CNN 백본(ResNet 등)과 결합된 Q-Network (MainNet, TargetNet) 구조
│   ├── agent.py                # DDQN 에이전트 클래스, 행동 선택(ε-greedy) 및 네트워크 업데이트 로직
│   ├── replay_buffer.py        # 경험 재생 버퍼 (Prioritized Experience Replay - PER 구현)
│   ├── environment.py          # 강화학습 환경 구축 (상태 관찰, 정답 비교, 적응형 보상 함수 구현)
│   └── utils.py                # 클래스 가중치(γ_k) 계산, 로깅, 시드(Seed) 고정 등 유틸리티 함수
│
├── train.py                    # 에이전트 학습을 실행하는 메인 스크립트 (Training Loop)
├── evaluate.py                 # 학습된 모델의 성능 평가 (F1-score, G-mean 계산 및 테스트)
│
├── checkpoints/                # 학습 중 저장되는 모델 가중치(.pth) 폴더
│   └── best_model.pth          # 가장 성능이 좋은 모델 파일
│
├── notebooks/                  # EDA 및 실험용 주피터 노트북 폴더
│   └── 01_data_exploration.ipynb # 데이터 분포 확인 및 MONAI 전처리 테스트용 노트북
│
├── requirements.txt            # 프로젝트 실행에 필요한 패키지 목록 (PyTorch, MONAI, Pandas 등)
└── README.md                   # 프로젝트 개요, 파이프라인 설명 및 실행 방법 안내
