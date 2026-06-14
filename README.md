# 하드웨어 가변 파라미터 변동에 따른 모델간 성능 비교

> ## MOT 17, 20 기반 Tracking Project
> 
> 본 프로젝트는 다중 객체 추적(MOT) 환경에서 CNN, Transformer, Lightweight 3대 아키텍처 패러다임 모델을 구축하고, 시나리오에 따라 실시간성(FPS), 메모리 점유율(Peak VRAM), 그리고 추적 정확도(MOTA, ID Swaps) 간의 트레이드오프(Trade-off)를 정량적으로 비교 분석하는 통합 실험 프레임워크입니다.

---

## 프로젝트 핵심 특징 (Key Features)

1. **모델 리스트**
   - **CNN**: YOLOv11 (Ultralytics + ByteTrack)
   - **Transformer**: RT-DETR (Object Queries 기반 엔드투엔드 모델)
   - **Lightweight**: 순정 MobileNet-SSD 백본에 **Scipy 고속 선형 할당(Hungarian Algorithm)** 추적 매칭 엔진을 직접 설계 및 이식하여 프레임워크 한계 극복 및 대조군 밸런스 확보
2. **심볼릭 링크(Symbolic Link) 데이터 파티셔닝**
   - 대용량 MOT17 / MOT20 원본 데이터를 심볼릭 링크를 활용해 스토리지 부하를 최소화하고 빠른 속도로 변환하도록 구성
3. **하이퍼 파리미터 자동 측정**
   - 다양한 Learning Rate와 Optimizer 조합을 순회하며 파인튜닝을 수행, 검증 성능(mAP50)이 높은 모델을 선별
4. **3단계 멀티 로깅 구성**
   - 프레임 레벨 진행률(tqdm) ➔ 비디오 레벨 개별 지표(MOTA, ID Swaps) 정산 ➔ 시나리오별 마스터 CSV 리포트 자동 생성까지 3단계 로그 구현

---

## 프로젝트 디렉토리 구조 (Architecture)

```text
AI_human_tracking/
├── config/
│   └── scenarios.yaml          # 영상 파일 파라미터 시나리오 설정 파일
├── data/
│   ├── MOT17/                  # 원본 다운로드 데이터셋 (train/test)
│   ├── MOT20/                  # 원본 다운로드 데이터셋 (train/test)
│   ├── processed_mot/          # 전처리 된 데이터 셋
│   │   ├── detection/          # Train / Val 이미지 및 YOLO 규격 .txt 라벨 (심볼릭)
│   │   └── tracking/           # Test 목적의 오리지널 시퀀스 스트림 데이터
│   └── mot_transfer.yaml       # Ultralytics 파인튜닝용 데이터 라우팅 매핑 파일
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ultralytics_tracker.py
│   │   └── mobilenet_tracker.py # Scipy 헝가리안 알고리즘 매칭 기법 내장 트래커
│   └── utils/
│       ├── __init__.py
│       └── metrics.py          # FPS/Peak VRAM 측정 유틸
├── weights/                    # 최적 모델 및 하이퍼파라미터 튜닝 결과 저장
│   ├── yolo11_best.pt          # 자동 선별된 최적 YOLO 가중치
│   └── rtdetr_best.pt          # 자동 선별된 최적 RT-DETR 가중치
├── convert_mot_to_yolo.py      # MOT17+20 통합 심볼릭 고속 변환 및 영상 단위 3분할 스크립트
├── train.py                    # 최적 가중치 양산용 자동화 파인튜닝 스윕 제어 통제실
├── main.py                     # [MAIN] 40라운드 종합 교차 벤치마크 마스터 스케줄러
└── requirements.txt            # 환경 의존성 환경 명세서
```

---

## 5대 가변 실험 시나리오 명세 (`scenarios.yaml`)

| 시나리오 ID              | 시나리오명     | 핵심 엔지니어링 목적                                | 이미지 제어 파라미터                               |
|:---------------------|:----------|:-------------------------------------------|:------------------------------------------|
| **0_Baseline**       | 표준 성능     | 모든 테스트 모델의 성능 기준 환경                        | `imgsz: 640`, `conf: 0.25`, `buffer: 30`  |
| **1_Fast_Light**     | 고속/경량화 지향 | 해상도 다운사이징을 통한 고속 처리 테스트 환경                 | `imgsz: 320`, `conf: 0.25`, `buffer: 30`  |
| **2_High_Precision** | 정밀 추적 지향  | 입력 해상도 상향 & 임계값 제어를 통한 미탐지율(FN) 테스트 환경     | `imgsz: 1280`, `conf: 0.50`, `buffer: 30` |
| **3_Tenacious**      | 끈질긴 추적 지향 | 임계값 최하향 및 버퍼 극대화로 겹침(Occlusion) 트래킹 테스트 환경 | `imgsz: 640`, `conf: 0.10`, `buffer: 64`  |
| **4_Strict**         | 엄격한 추적 지향 | 오탐지율(FP) 테스트 환경                            | `imgsz: 640`, `conf: 0.40`, `buffer: 15`  |

---

## 시작 가이드 (Quick Start)

가상환경 구축, CUDA 12.8 대응 하드웨어 가속 PyTorch 빌드 선행 설치, 그리고 대용량 MOT 데이터셋 다운로드 및 물리 배치까지 단 한 줄의 명령어로 처리할 수 있는 올인원 셸 스크립트를 사용함.

### 0. 깃허브 클론
```bash
git clone https://github.com/ksj20024/AI_human_tracking.git
```

### 1. 통합 환경 셋업
프로젝트 최상위 경로에서 아래 쉘 스크립트 실행
```bash
# 1) 스크립트에 실행 권한 부여
chmod +x setup_all.sh

# 2) 통합 환경 생성 스크립트 실행
./setup_all.sh
```
* **주의**: 간혹 motchallenge 서버의 문제 또는 스크립트 파일 오류로 데이터 파일 셋업이 안되는 경우에 대비해 kaggle을 통한 데이터 다운로드 방법

```bash
# 1. 혹시 모를 kagglehub 설치 (콘다 환경 내부 주입)
pip install kagglehub

# 2. 스크립트 실행
python utils/download_datasets.py
```

### 2. 하이퍼파라미터 자동 튜닝 및 최적 가중치 측정
그리드 스페이스 스케줄러를 순회하며 전이 학습을 수행하고, 검증 성능(mAP50)이 가장 높은 각 모델별 최적 가중치를 `./weights`에 저장
```bash
python train.py
또는
nohup python train.py > train.log 2>&1 &
```

### 3. 테스트 런타임 실행
미리 지정한 시나리오 별 인풋값에 따라 이미지를 변환해 모델 간의 교차 벤치마크 데이터를 생성. 연산이 완료되면 `./results` 폴더에 CSV 형태로 저장.
```bash
python main.py
또는
nohup python main.py > test.log 2>&1 &
```

---

## 평가 지표 및 산출 결과물 (Outputs)

테스트 종료시 콘솔에 마스터 표가 출력되며, `./results/complete_benchmark_report.csv` 파일에 저장

* **하드웨어 효율성 지표**: `FPS` (실시간 처리 속도), `Peak_VRAM_GB` (최대 그래픽 메모리 점유율)
* **글로벌 표준 추적 정확도 지표**: `MOTA_Percent` (종합 추적 정확도), `ID_Swaps` (고유 아이디 유실/꼬임 횟수), `False_Positives_FP` (오탐지 박스 수)
