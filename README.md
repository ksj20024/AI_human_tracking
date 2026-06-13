# 🚀 하드웨어 가변 파라미터 변동에 따른 3대 딥러닝 패러다임 MOT 정량 벤치마크 연구

> **학과/졸업 프로젝트 (Human Tracking Pipeline Benchmarking)**
> 본 연구 프로젝트는 다중 객체 추적(MOT) 환경에서 CNN, Transformer, Lightweight 3대 아키텍처 패러다임 모델을 구축하고, 하드웨어 가변 파라미터 5대 시나리오에 따른 실시간성(FPS), 메모리 점유율(Peak VRAM), 그리고 추적 정확도(MOTA, ID Swaps) 간의 트레이드오프(Trade-off)를 정량적으로 비교 분석하는 통합 실험 프레임워크입니다.

---

## 🌟 프로젝트 핵심 특장점 (Key Features)

1. **3대 정예 패러다임 라인업 구축**
   - **CNN 대표**: YOLOv11 (Ultralytics + ByteTrack)
   - **Transformer 대표**: RT-DETR (Object Queries 기반 엔드투엔드 모델)
   - **Lightweight 대표**: 순정 MobileNet-SSD 백본에 **Scipy 고속 선형 할당(Hungarian Algorithm)** 추적 매칭 엔진을 직접 설계 및 이식하여 프레임워크 한계 극복 및 대조군 밸런스 확보
2. **초고속 심볼릭 링크(Symbolic Link) 데이터 파티셔닝**
   - 대용량 MOT17 / MOT20 원본 데이터를 물리적으로 복사하지 않고, 심볼릭 링크를 활용해 디스크 I/O 및 EBS 스토리지 용량 부하를 0초 만에 해결하는 고속 하이브리드 변환기 내장
3. **자동화 하이퍼파라미터 그리드 스윕(Grid Sweep)**
   - 다양한 Learning Rate와 Optimizer 조합을 순회하며 파인튜닝을 수행하고, 검증 성능(mAP50)이 가장 높은 챔피언 모델을 자동 선별하는 제어 세션 완비
4. **3단계 멀티 그레뉴얼러티 로깅 및 무결점 정적 타입 추론**
   - 프레임 레벨 진행률(tqdm) ➔ 비디오 레벨 개별 지표(MOTA, ID Swaps) 정산 ➔ 시나리오별 마스터 CSV 리포트 자동 생성까지 3단계 로그 구현
   - PyCharm 정적 타입 분석기(Linter)의 경고등을 원천 차단한 무결점 Clean Code 달성

---

## 📂 프로젝트 디렉토리 구조 (Architecture)

```text
AI_human_tracking/
├── config/
│   └── scenarios.yaml          # 5대 가변 하드웨어 파라미터 시나리오 설정 파일
├── data/
│   ├── MOT17/                  # 원본 다운로드 데이터셋 (train/test)
│   ├── MOT20/                  # 원본 다운로드 데이터셋 (train/test)
│   ├── processed_mot/          # [자동 생성] 하이브리드 고속 가공 데이터셋
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
│       └── metrics.py          # torch.cuda 정밀 동기화 기반 FPS/Peak VRAM 측정 유틸
├── weights/                    # 챔피언 모델 및 하이퍼파라미터 튜닝 리포트 보관소
│   ├── yolo11_best.pt          # 자동 선별된 최적 YOLO 가중치
│   └── rtdetr_best.pt          # 자동 선별된 최적 RT-DETR 가중치
├── convert_mot_to_yolo.py      # MOT17+20 통합 심볼릭 고속 변환 및 영상 단위 3분할 스크립트
├── train.py                    # 최적 가중치 양산용 자동화 파인튜닝 스윕 제어 통제실
├── main.py                     # [MAIN] 40라운드 종합 교차 벤치마크 마스터 스케줄러
└── requirements.txt            # 환경 의존성 환경 명세서
```

---

## ⚡ 5대 가변 실험 시나리오 명세 (`scenarios.yaml`)

| 시나리오 ID | 시나리오명 | 핵심 엔지니어링 목적 | 주요 하드웨어 제어 파라미터 |
| :--- | :--- | :--- | :--- |
| **0_Baseline** | 표준 성능 세팅 | 모든 실험 모델의 하드웨어 연산 기준점 수립 | `imgsz: 640`, `conf: 0.25`, `buffer: 30` |
| **1_Fast_Light** | 고속/경량화 지향 | 해상도 다운사이징을 통한 초고속 엣지 환경 시뮬레이션 | `imgsz: 320`, `conf: 0.25`, `buffer: 30` |
| **2_High_Precision**| 고정밀 추적 지향 | 입력 해상도 상향 및 임계값 제어로 미탐지율(FN) 극소화 | `imgsz: 1280`, `conf: 0.50`, `buffer: 30` |
| **3_Tenacious** | 끈질긴 추적 지향 | 임계값 최하향 및 버퍼 극대화로 가려짐(Occlusion) 극복 | `imgsz: 640`, `conf: 0.10`, `buffer: 64` |
| **4_Strict** | 엄격한 추적 지향 | 오탐지율(FP)을 극소화하기 위한 타이트한 필터링 세팅 | `imgsz: 640`, `conf: 0.40`, `buffer: 15` |

---

## 🚀 시작 가이드 (Quick Start)

### 1. 가상환경 구축 및 환경 설정
본 프레임워크는 GPU 아키텍처 가속(CUDA) 인식을 위해 아래 설치 순서를 엄격히 준수해야 합니다.
```bash
# CUDA 12.6 대응 PyTorch 및 비전 툴킷 선행 설치
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126) --force-reinstall

# 3대 패러다임 모델 및 헝가리안 매칭 종속 패키지 일괄 주입
pip install -r requirements.txt
```

### 2. 고속 심볼릭 데이터 변환 실행
원본 데이터셋(`data/MOT17`, `data/MOT20`)을 배치한 후, 아래 스크립트를 실행하여 복사 오버헤드 없는 하이브리드 3분할 데이터 바구니를 빌드합니다.
```bash
python convert_mot_to_yolo.py
```

### 3. 하이퍼파라미터 자동 튜닝 및 최적 가중치 구워내기
그리드 스페이스를 자동으로 순회하며 전이 학습을 수행하고 최고의 가중치를 `./weights`에 고정합니다.
```bash
python train.py
```

### 4. 종합 벤치마크 런타임 가동
최종 지휘 통제실을 트리거하여 5대 시나리오 데이터 양산을 개시합니다.
```bash
python main.py
```

---

## 📊 평가 지표 및 산출 결과물 (Outputs)

본 파이프라인 연산이 끝나면 콘솔에 정렬된 마스터 표가 출력되며, 즉시 `./results/complete_benchmark_report.csv` 파일로 하드디스크에 영구 정산됩니다. 보고서 및 논문 작성 시 해당 CSV 수치를 인용하십시오.

* **하드웨어 효율성 지표**: `FPS` (실시간 처리 속도), `Peak_VRAM_GB` (최대 그래픽 메모리 점유율)
* **글로벌 표준 추적 정확도 지표**: `MOTA_Percent` (종합 추적 정확도), `ID_Swaps` (고유 아이디 유실/꼬임 횟수), `False_Positives_FP` (오탐지 박스 수)