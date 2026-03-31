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