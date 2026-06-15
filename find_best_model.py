import os
import shutil
import pandas as pd


def locate_and_deploy_champion(project_dir, target_output_path):
    """모든 스윕 폴더의 results.csv를 파싱하여 가장 높은 mAP50을 기록한 가중치를 추출합니다."""
    if not os.path.exists(project_dir):
        print(f"ℹ️ 아직 학습 전이거나 폴더가 없습니다: {project_dir}")
        return

    absolute_best_map = -1.0
    champion_folder = None

    # 1. 모든 하위 폴더를 돌며 results.csv 탐색
    for root, dirs, files in os.walk(project_dir):
        if "results.csv" in files:
            csv_path = os.path.join(root, "results.csv")
            try:
                df = pd.read_csv(csv_path)
                # 컬럼 이름 양끝 공백 제거
                df.columns = [c.strip() for c in df.columns]

                # YOLO/DETR 규격 mAP50 컬럼 매칭 ('metrics/mAP50(B)' 또는 'val/mAP50' 등)
                map_col = [c for c in df.columns if 'mAP50' in c and '95' not in c]

                if map_col:
                    current_max_map = df[map_col[0]].max()
                    if current_max_map > absolute_best_map:
                        absolute_best_map = current_max_map
                        champion_folder = root
            except Exception as e:
                print(f"⚠️ {csv_path} 읽기 실패 (스킵): {e}")

    # 2. 찾아낸 진짜 챔피언 가중치를 main.py가 보는 weights/ 폴더로 복사
    if champion_folder:
        print(f"👑 [{project_dir}]의 진짜 챔피언 발견: {os.path.basename(champion_folder)}")
        print(f"   ➔ 최고 성적 (mAP50): {absolute_best_map * 100:.2f}%")

        weights_dir = os.path.join(champion_folder, "weights")
        src_pt = os.path.join(weights_dir, "best.pt")

        # 만약 강종되어 best.pt가 없다면 last.pt라도 구제
        if not os.path.exists(src_pt):
            src_pt = os.path.join(weights_dir, "last.pt")

        if os.path.exists(src_pt):
            os.makedirs(os.path.dirname(target_output_path), exist_ok=True)
            shutil.copy(src_pt, target_output_path)
            print(f"✅ 가중치 배달 완료: {src_pt} ➔ {target_output_path}\n")
        else:
            print(f"❌ 폴더는 찾았으나 가중치 파일이 유실되었습니다: {weights_dir}\n")
    else:
        print(f"❌ [{project_dir}] 내에 유효한 학습 기록(results.csv)이 없습니다.\n")


if __name__ == "__main__":
    print("=========================================================")
    print("🔍 파편화된 튜닝 결과물 정밀 정산 및 챔피언 가중치 수집 가동")
    print("=========================================================\n")

    # YOLO 결과물 정산
    locate_and_deploy_champion("runs_tuning_yolo", "weights/yolo11_best.pt")

    # RT-DETR 결과물 정산
    locate_and_deploy_champion("runs_tuning_detr", "weights/rtdetr_best.pt")