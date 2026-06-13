import os
import shutil
from pathlib import Path

print("🔍 kagglehub 라이브러리 확인 중...")
try:
    import kagglehub
except ImportError:
    print("📦 kagglehub가 없습니다. 설치를 진행합니다.")
    os.system("pip install kagglehub")
    import kagglehub


def setup_kaggle_dataset(kaggle_repo: str, target_dir_name: str):
    print("=" * 60)
    print(f"🚀 Kaggle에서 {target_dir_name} 다운로드 스케줄러 가동")
    print(f"🔗 레포지토리: {kaggle_repo}")
    print("=" * 60)

    # 1. 캐그랩을 통해 다운로드 (자동 압축 해제까지 수행됨)
    downloaded_cache_path = kagglehub.dataset_download(kaggle_repo)
    print(f"✅ 다운로드 완료 (시스템 캐시 경로: {downloaded_cache_path})")

    # 2. 우리 프로젝트 내부의 목적지 경로 설정 (./data/MOT17 등)
    project_target_path = Path(f"./data/{target_dir_name}")
    project_target_path.parent.mkdir(parents=True, exist_ok=True)

    # 기존에 잘못 깔렸거나 비어있는 폴더가 있다면 깔끔하게 밀고 시작
    if project_target_path.exists():
        shutil.rmtree(project_target_path)

    # 3. 캐시 폴더의 내용물을 프로젝트 data 폴더로 이동
    print(f"🚚 캐시 저장소 ➔ 프로젝트 폴더({project_target_path})로 데이터 동기화 중...")
    shutil.copytree(downloaded_cache_path, project_target_path)

    # 💡 [보정] 중복 폴더 구조 해결 (data/MOT17/MOT17 -> data/MOT17)
    nested_path = project_target_path / target_dir_name
    if nested_path.is_dir():
        print(f"🔄 중복 폴더 구조 감지 ({nested_path}) ➔ 구조 평탄화 진행 중...")
        for file_path in nested_path.iterdir():
            # 알맹이들을 한 단계 상위 폴더(project_target_path)로 이동
            shutil.move(str(file_path), str(project_target_path))
        # 알맹이가 빠져나간 빈 중복 폴더 삭제
        nested_path.rmdir()

    print(f"✨ {target_dir_name} 인프라 배치 완료!\n")


if __name__ == "__main__":
    # 사용자가 제시한 무결점 캐글 데이터셋 핸들러 매핑
    setup_kaggle_dataset("wenhoujinjust/mot-17", "MOT17")
    setup_kaggle_dataset("ismailelbouknify/mot-20", "MOT20")

    print("==================================================================")
    print("🎉 모든 데이터셋이 프로젝트 내 data/ 폴더에 완벽하게 안착했습니다!")
    print("💡 다음 단계인 'python convert_mot_to_yolo.py'를 즉시 실행하세요.")
    print("==================================================================")