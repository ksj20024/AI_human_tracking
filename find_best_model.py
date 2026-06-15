import shutil
from pathlib import Path
from typing import Any  # 👈 Any 임포트 추가
import pandas as pd


def locate_and_deploy_champion(project_dir: str, target_output_path: str) -> None:
    """모든 스윕 폴더의 results.csv를 파싱하여 가장 높은 mAP50을 기록한 가중치를 추출합니다.

    Pathlib과 명시적 Any 캐스팅을 적용하여 파이참의 판다스 타입 경고를 완벽히 격파합니다.
    """
    base_path = Path(project_dir)
    if not base_path.exists():
        print(f"ℹ️ 아직 학습 전이거나 폴더가 없습니다: {project_dir}")
        return

    absolute_best_map = -1.0
    champion_folder: Path | None = None

    for csv_path in base_path.rglob("results.csv"):
        try:
            raw_data = pd.read_csv(csv_path)

            if not isinstance(raw_data, pd.DataFrame):
                continue

            df: pd.DataFrame = raw_data
            df.columns = [str(c).strip() for c in df.columns]

            map_col = [c for c in df.columns if "mAP50" in c and "95" not in c]

            if map_col:
                # 💡 [치트키] 판다스의 지옥 같은 20개 유니온 타입을 Any로 덮어씌워 무력화합니다.
                # 이렇게 하면 파이참이 더 이상 내부 타입을 추적하지 않아 군더더기 경고가 싹 사라집니다.
                raw_max: Any = df[map_col[0]].max()

                if hasattr(raw_max, "max"):
                    current_max_map = float(raw_max.max())
                else:
                    current_max_map = float(raw_max)

                if current_max_map > absolute_best_map:
                    absolute_best_map = current_max_map
                    champion_folder = csv_path.parent

        except Exception as e:
            print(f"⚠️ {csv_path} 읽기 실패 (스킵): {e}")

    if champion_folder:
        print(f"👑 [{project_dir}]의 진짜 챔피언 발견: {champion_folder.name}")
        print(f"   ➔ 최고 성적 (mAP50): {absolute_best_map * 100:.2f}%")

        weights_dir = champion_folder / "weights"
        src_pt = weights_dir / "best.pt"

        if not src_pt.exists():
            src_pt = weights_dir / "last.pt"

        if src_pt.exists():
            target_path = Path(target_output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(str(src_pt), str(target_path))
            print(f"✅ 가중치 배달 완료: {src_pt.name} ➔ {target_output_path}\n")
        else:
            print(f"❌ 폴더는 찾았으나 가중치 파일이 유실되었습니다: {weights_dir}\n")
    else:
        print(f"❌ [{project_dir}] 내에 유효한 학습 기록(results.csv)이 없습니다.\n")


if __name__ == "__main__":
    print("=========================================================")
    print("🔍 파편화된 튜닝 결과물 정밀 정산 및 챔피언 가중치 수집 가동")
    print("=========================================================\n")

    locate_and_deploy_champion("./runs/detect/runs_tuning_yolo", "weights/yolo11_best.pt")
    locate_and_deploy_champion("./runs/detect/runs_tuning_detr", "weights/rtdetr_best.pt")