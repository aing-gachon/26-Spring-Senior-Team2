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