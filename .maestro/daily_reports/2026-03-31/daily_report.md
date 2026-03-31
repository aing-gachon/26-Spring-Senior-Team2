# 🗓️ Daily Report - 2026년 3월 31일

---
## [1차 세션 (이전 기록 복구)]

### 🕒 목표
- 프로젝트 모듈 분리 및 강화학습 뼈대 구축 (MedicalImageEnv, DDQN Agent 연동)

### 🛠 수정 내용
- 15개 질병을 취급하던 다중 분류를 특정 1개(Nodule 등) vs 정상 모델로 스코프 축소.
- MONAI 기반 파이프라인 생성 및 `configs/config.yaml` 중앙 통제소 설립.
- DataFrame(CSV)를 이용한 Image Loader 및 1:50 데이터 비율 추출 함수 구현.
- Double DQN + Prioritized Replay Buffer 메모리 스크립트 작성.

---

## [2차 세션 (최신 업데이트 내역)]

### 🕒 오늘의 목표 (Objective)
- BicDRL 논문의 설계 의도를 완전히 분해하여, 기존 파이프라인의 **보상, 평가, 하이퍼파라미터, 전처리 체계**를 논문의 실제 은닉된 실험 환경과 100% 동기화하고 최초의 터미널 훈련 실행(Test Run)을 성공시킨다.

### 🛠 수정된 내부 아키텍처 (Modified Files)
1. **`src/data/dataset.py` & `data/raw/` 폴더 구조화**
   - 방대한 CSV 파일을 뒤지는 구시대적 방식 폐기. 직관적인 `Nodule`과 `No Finding` ImageFolder 스로틀 방식을 도입하여 작업 난이도를 대폭 하락.
2. **`configs/config.yaml` 튜닝**
   - 테스트를 위한 에피소드 극단적 축소(5000 ➔ 5)
   - 저자 실제 세팅값 일치화 완료 (학습률 `0.00025`, 할인율 `0.1`, 배치 `64`)
3. **`src/rl/environment.py` 보상 체계 개편**
   - 극심한 불균형(1:50) 극복을 위한 Nodule(+50), Normal(+1) 차등 보상공식 신규 적용.
4. **`src/data/transforms.py` 강화**
   - 흑백 사진용 3채널 병합 기능 확충 (ResNet50 호환)
5. **`evaluate.py` 평가 고도화**
   - 단순 F1-score 뿐만 아니라 모델의 실제 임상 성능을 증명할 `G-Mean(Sens * Spec)` 도출식 개발.
6. **`Docs/DDQN_Architecture_DeepDive.md` (NEW)**
   - 논문에 숨겨진 백본 CNN+FC 하이브리드 투트랙 모델의 설계 당위성을 문서화.

### 🚨 문제 발생 및 트러블슈팅 (Issues & Troubleshooting)
- **에러 1**: `1 Channel vs 3 Channel (ResNet50)` -> `transforms.py`에 `RepeatChanneld(3)` 긴급 투입하여 스케일링 방어 및 해결.
- **에러 2**: `ImportError` -> `dataset.py`에서 삭제된 함수를 부르던 `__init__.py` 청소 완료.
- **현상 3**: `Loss 값이 0.0` -> 고장이 아닌 **초기 경험 결핍에 따른 정상 대기 현상**. 버퍼 정원(64개) 초과 시점부터 연산 재개 검증.

### 🎯 다음 진행 예정 사항 (Next Milestones)
- **Google Colab (NVIDIA T4 GPU)** 환경을 통해 코드를 연동시키고, 실제 논문 기재 수치인 5000 Episodes 실전 마라톤 완전 훈련 가동 예정.
