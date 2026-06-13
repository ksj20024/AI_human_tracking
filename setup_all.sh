#!/bin/bash

# 실행 중 에러가 발생하면 즉시 중단하는 안전장치
set -e

echo "=================================================================="
echo "파이썬 라이브러리 + MOT 데이터셋 통합 빌드 가동"
echo "=================================================================="

# 1. 현재 활성화된 Conda 환경 정보 출력 (디버깅용 로그)
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "현재 활성화된 Conda 환경 감지됨 : [$CONDA_DEFAULT_ENV]"
else
    echo "활성화된 Conda 환경 변수가 보이지 않습니다. 현재 터미널 환경에 직접 설치를 진행합니다."
fi

# 2. 현재 환경의 pip 최신화 및 CUDA 12.8 대응 PyTorch 선행 설치
echo "CUDA 12.8 대응 PyTorch 및 핵심 비전 툴킷 설치 중..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# 3. 종속성 패키지 일괄 주입
if [ -f "requirements.txt" ]; then
    echo "requirements.txt 기반 의존성 패키지 일괄 설치 중..."
    pip install -r requirements.txt
else
    echo "requirements.txt 파일이 보이지 않아 필수 패키지를 개별 설치합니다."
    pip install ultralytics opencv-python motmetrics scipy numpy pandas pyyaml tqdm
fi

echo "파이썬 실행 환경 세팅 완료"
echo "------------------------------------------------------------------"

# 4. 데이터 폴더 구조 미리 생성
echo " 데이터 디렉토리 구조 검증 및 생성..."
mkdir -p data/MOT17 data/MOT20

# 5. MOT17 백그라운드 고속 다운로드 및 압축 해제 후 삭제
echo "MOT17 다운로드 및 배치 시작 (대용량)..."
wget -q --show-progress -O data/MOT17.zip https://motchallenge.net/data/MOT17.zip
echo "MOT17 압축 해제 중..."
unzip -q data/MOT17.zip -d data/MOT17
rm data/MOT17.zip
echo "MOT17 배치 완료"
echo "------------------------------------------------------------------"

# 6. MOT20 백그라운드 고속 다운로드 및 압축 해제 후 삭제
echo "MOT20 다운로드 및 배치 시작 (대용량)..."
wget -q --show-progress -O data/MOT20.zip https://motchallenge.net/data/MOT20.zip
echo "MOT20 압축 해제 중..."
unzip -q data/MOT20.zip -d data/MOT20
rm data/MOT20.zip
echo "MOT20 배치 완료"

echo "=================================================================="
echo "환경 세팅 및 데이터셋 배치 전 과정 완료"
echo "다음 단계: 'python convert_mot_to_yolo.py'를 가동하세요."
echo "=================================================================="