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
