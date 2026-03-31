# 📈 프로젝트 진행 단계 요약 (Progress Status)

| 진행 상태 | 모듈 / 단계 | 상세 작업 내용 |
|:---:|---|---|
| ✅ 완료 | **Step 1: 환경 설정** | `requirements.txt` 세팅, 범용 유틸(`utils.py`) 및 하이퍼파라미터(`config.yaml`) 뼈대 구성 완료 |
| ✅ 완료 | **Step 2: 데이터 파이프라인** | CSV 단일 라벨 필터링, MONAI Dict Transform 기반 DataLoader 생성 및 오류 방지(교집합) 로직 적용 완료 |
| ✅ 완료 | **Step 3: AI 아키텍처** | ResNet50 특징 추출기 백본(Feature Extractor) 및 행동가치 판단을 위한 Q-Network 결합 헤드 생성 |
| ✅ 완료 | **Step 4: RL 핵심 모듈화** | `Gymnasium` 환경 클래스 생성, DDQN 기반 에이전트, TD-Error 기반 PER 버퍼 적용 완료 |
| ✅ 완료 | **Step 5: 메인 스크립트** | 훈련(Train) 에피소드 루프 스크립트 및 성능 검증(Evaluate) 스크립트 작성 완료 |

---

## 🚀 앞으로의 마일스톤 (To-Do / Roadmap)
1. **[Data] 데이터 파일 마운트**: 실제 Kaggle / NIH 엑스레이 데이터셋 다운로드 후 `data/raw` 폴더에 분류하여 삽입.
2. **[Test] 에이전트 최초 훈련 개시**: 터미널에서 `train.py`를 최초 실행하고 GPU 자원 소모, 에러 발생 여부 및 Reward 등락폭 눈으로 모니터링하기.
3. **[Tune] 하이퍼파라미터 튜닝**: Epsilon Decay 수치 갱신, Replay Buffer 크기 조정 등을 통한 학습 속도 보정.
