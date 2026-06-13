import os
import configparser
import sys
from pathlib import Path


def convert_mot_to_yolo_core(seq_dir: Path, output_base_path: Path, target_split: str):
    """
    사용자가 구축한 고속 심볼릭 링크 및 YOLO 정규화 변환 코어 로직
    """
    img_out_dir = output_base_path / 'detection' / 'images' / target_split
    lbl_out_dir = output_base_path / 'detection' / 'labels' / target_split

    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 영상 해상도 정보 파싱 (seqinfo.ini)
    ini_path = seq_dir / 'seqinfo.ini'
    if not ini_path.exists():
        return

    config = configparser.ConfigParser()
    config.read(ini_path)
    img_width = int(config['Sequence']['imWidth'])
    img_height = int(config['Sequence']['imHeight'])

    gt_txt = seq_dir / 'gt' / 'gt.txt'
    if not gt_txt.exists():
        return

    print(f"🚀 [SYM-LINK CONVERT] {seq_dir.name} ➔ [{target_split.upper()} SET] 가공 중...")

    # 2. gt.txt 읽어서 프레임별로 레이블 그룹화
    frame_labels = {}
    with open(gt_txt, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue

            frame_id = int(parts[0])
            obj_class = int(parts[7])  # 1: Pedestrian (보행자)

            if obj_class != 1:
                continue

            x_min = float(parts[2])
            y_min = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            # 정규화 및 중앙값 환산
            x_center = (x_min + w / 2.0) / img_width
            y_center = (y_min + h / 2.0) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # 경계값 0.0 ~ 1.0 제한 예외 처리
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            if frame_id not in frame_labels:
                frame_labels[frame_id] = []

            frame_labels[frame_id].append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # 3. 이미지 파일 루프 돌며 고속 심볼릭 링크 생성
    img_dir = seq_dir / 'img1'
    for img_path in img_dir.glob('*.jpg'):
        frame_id = int(img_path.stem)

        new_img_name = f"{seq_dir.name}_{img_path.name}"
        new_img_path = img_out_dir / new_img_name
        new_lbl_path = lbl_out_dir / f"{seq_dir.name}_{img_path.stem}.txt"

        # EBS I/O 부하 0초 최적화 링크 연결
        if not new_img_path.exists():
            os.symlink(img_path.resolve(), new_img_path)

        labels = frame_labels.get(frame_id, [])
        with open(new_lbl_path, 'w') as lf:
            lf.write('\n'.join(labels))


if __name__ == "__main__":
    # 💡 [핵심 수정] 파일 위치(src/utils/현재파일)를 기준으로 프로젝트 최상위 루트 찾기
    # parents[0] = utils, parents[1] = src, parents[2] = 프로젝트 루트
    # 1. 파일 위치 기준으로 프로젝트 최상위 루트 역산
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    print("==================================================================")
    print(f"🔍 [경로 검증] 계산된 프로젝트 루트: {PROJECT_ROOT}")
    print("==================================================================")

    # 2. 🔥 뇌절 방지용 안전장치 (Assertion)
    # 루트 폴더에 반드시 있어야 하는 'data' 폴더가 없는 엉뚱한 위치라면 즉시 실행 차단!
    if not (PROJECT_ROOT / "data").exists():
        print(f"❌ [경로 에러] 예상한 루트에 'data' 폴더가 없습니다!")
        print(f"👉 현재 계산된 위치: {PROJECT_ROOT}")
        print(f"🛠️ 해결책: 이 스크립트가 'src/utils/' 폴더 안에 정상적으로 위치해 있는지 확인하세요.")
        sys.exit(1)  # 프로그램 강제 종료
    else:
        print("✅ [검증 통과] 'data' 폴더가 정상 확인되었습니다. 진격을 개시합니다.")
        print("==================================================================")

    # 소스 데이터셋 루트 디렉토리 정의 (이제 PROJECT_ROOT는 무조건 무결함)
    SRC_DATASET_ROOTS = [PROJECT_ROOT / "data" / "MOT17", PROJECT_ROOT / "data" / "MOT20"]
    OUTPUT_BASE_DIR = PROJECT_ROOT / "data" / "processed_mot"

    # MOT17 및 MOT20의 시퀀스 단위 하이브리드 분할 매핑 명세 수립
    TRAIN_KEYWORDS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT20-01", "MOT20-02"]
    VAL_KEYWORDS = ["MOT17-10", "MOT17-11", "MOT20-03"]
    TEST_KEYWORDS = ["MOT17-13", "MOT20-05"]

    print("==================================================================")
    print("⚡ [INTEGRATED DATA LOG] MOT17 + MOT20 하이브리드 심볼릭 변환 개시")
    print("==================================================================")

    for src_root in SRC_DATASET_ROOTS:
        raw_train_path = src_root / "train"
        if not raw_train_path.exists():
            print(f"⚠️ 데이터셋 경로가 존재하지 않아 스킵합니다: {raw_train_path}")
            continue

        print(f"\n📦 타겟 데이터셋 파싱 중 ➔ {src_root.name}")

        for seq_dir in sorted(raw_train_path.iterdir()):
            if not seq_dir.is_dir():
                continue

            is_processed = False

            # Train 분기 체크
            if any(kwd in seq_dir.name for kwd in TRAIN_KEYWORDS):
                convert_mot_to_yolo_core(seq_dir, OUTPUT_BASE_DIR, target_split='train')
                is_processed = True

            # Val 분기 체크
            elif any(kwd in seq_dir.name for kwd in VAL_KEYWORDS):
                convert_mot_to_yolo_core(seq_dir, OUTPUT_BASE_DIR, target_split='val')
                is_processed = True

            # Test 분기 체크
            elif any(kwd in seq_dir.name for kwd in TEST_KEYWORDS):
                tracking_test_dst = OUTPUT_BASE_DIR / "tracking" / "test" / seq_dir.name
                tracking_test_dst.parent.mkdir(parents=True, exist_ok=True)

                if not tracking_test_dst.exists():
                    os.symlink(seq_dir.resolve(), tracking_test_dst)
                print(f"ℹ️ [TRACKING TEST ISOLATION] {seq_dir.name} ➔ STAGE 3 평가군 링크 완료.")
                is_processed = True

            if not is_processed:
                print(f"⚠️ [BYPASS] {seq_dir.name} 은 분할 정책에 명시되지 않아 안전하게 제외되었습니다.")

    print("\n" + "=" * 66)
    print(f"🎉 MOT17 & MOT20 최종 통합 변환 완료! ➔ 저장소: {OUTPUT_BASE_DIR}")
    print("=" * 66)