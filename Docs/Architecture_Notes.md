# 🧠 BicDRL-Reproduction 아키텍처 및 파이프라인 논의 노트

이 문서는 사용자가 구상한 DRL 기반 의료 이미지 처리 파이프라인과 그에 대한 리뷰 및 구조적 피드백을 기록한 문서입니다.

## 1. 초기 사용자 제안 파이프라인
1. **기본 구조**: `A_Deep_Reinforcement_Learning_Framework_for_Imbalanced_Medical_Image_Classification` 논문을 기반으로 RL을 보상함수 체계로 사용하여 의료 데이터 불균형에서 Multi-labeling의 성능을 올리는 방식을 채택.
2. **에이전트 입력 차별화**: 논문에서 Agent에게 어떤 방식으로 데이터를 정제해서 넘겼는지 명시되지 않음. 이에 따라 DDQN 방식을 통해 RL 보상을 진행하되, Agent에게 넘겨줄 데이터를 MONAI (CNN or Transformer)로 전처리하여 학습 기록을 따라가기로 결정.
3. **MONAI 전처리 단계**:
   - 차원 및 채널 조정 (`EnsureChannelFirstd`)
   - 크기 조정 및 자르기 (`Resized`, `CenterSpatialCropd`)
   - Dice Loss 방식을 차용한 전처리 시도 (장기 부분만 살리고 배경 제거)
   - 정규화 및 스케일링 (`ScaleIntensityd`)
   - 텐서 변환 (`EnsureTyped`)
4. **상태(State) 매핑**: `Data_Entry_2017_v2020.csv` 에서 질병 명을 One-Hot Encoding 하고, 전처리된 이미지와 합쳐 Agent의 입력 상태로 구성.
5. **학습**: 논문의 DDQN 보상 함수 체계 재현.

---

## 2. 주요 아키텍처 피드백 및 수정 방향

### 📌 1. Multi-labeling vs Multi-class 의 구조적 차이
- **논문의 방향**: N개의 클래스 중 하나를 행동(Action)으로 선택하는 이진/다중 클래스(Multi-class) MDP. 한 이미지 당 단일 질병(Single-finding) 혹은 이진 분류 환경 모델을 상정.
- **실제 데이터 적용**: 현재 NIH 데이터는 한 환자가 여러 질병을 갖는 Multi-label 데이터임. 논문 알고리즘을 그대로 적용하려면, 질병이 단 하나인 이미지만 필터링 하거나, 논문과 동일한 Binary Dataset을 사용해야 함.

### 📌 2. Dice Loss 적용에 대한 개념 수정
- Dice Loss는 의료 이미지 분할(Segmentation) 단계에서 예측과 정답 마스크를 비교하는 **손실(Loss) 함수** 임. 이를 변환(Transform) 파이프라인에 단순 적용할 수 없음.
- **대안**: 사전 학습된 분할(Segmentation) 모델로 폐 영역의 마스크를 생성하여 원본 이미지에 크롭(Crop) 후 넣을 수 있으나, 일반적으로 폐 주변 배경 정보도 질병 분류에 도움을 주므로 이미지 Resizing만 수행하는 것이 훨씬 안정적임.

### 📌 3. CNN / Transformer 모델의 활용 위치 (상세 논의 대상)
- **오류 방지**: 전처리된 텐서를 곧바로 평탄화(Flatten)시켜 RL Agent의 일반적인 선형(Linear) MainNet 에 입력하면 공간적 정보(Spatial Features)가 손실되고 파라미터가 과도하게 증가함.
- **해결책**: Agent의 Q-Network 본체를 ResNet, DenseNet 등 CNN 기반의 딥러닝 특징 추출기로 설계. 이미지가 CNN을 통과하여 나온 Feature map이나 HeatMap 등을 최종 Linear Layer를 통해 Q-Value로 변환하도록 완전 통합된 Network 구성이 필수.

### 📌 4. 상태(State) 구성 및 One-hot 벡터 사용 위치
- 의료 RL에서 에이전트의 **상태(State, $S_t$)**는 원칙적으로 입력 이미지 및 그 추출된 특징임.
- CSV에서 가져온 One-Hot 벡터는 Agent에게 입력되는 상태가 아니라, 에이전트가 행동을 취한 후 **정답(Target)과 비교해 보상(Reward)을 주기 위한 평가 기준표** 로 사용되어야 함.

---

## 3. 확정된 모델 스펙 사항
- **Backbone CNN**: ResNet 사용 예정
- **RL State 구성**: 순수 이미지만 넣는 것이 아니라, 이미지를 사전 CNN으로 분석하여 **HeatMap** 등 정보성 피처를 추출한 결과물과 이미지를 결합(또는 보강)하여 에이전트에 상태(State)로 전달.
