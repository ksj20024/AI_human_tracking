# main.py
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import motmetrics as mm
from tqdm import tqdm
from ultralytics import YOLO, RTDETR
from typing import List, Dict, Any
from src.utils.metrics import PerformanceTracker
from src.models.mobilenet_tracker import MobileNetSSDTrackerRunner, SimpleTrack, calculate_iou_matrix


def load_config(config_path="config/scenarios.yaml"):
    """5대 하이퍼파라미터 가변 시나리오 설정을 로드합니다."""
    import yaml
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_gt_for_tracking(seq_path: str) -> dict:
    """MOT 원본 gt.txt 파일을 파싱하여 프레임별 정답 딕셔너리를 구성합니다."""
    gt_data = {}
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        return gt_data

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                class_id = int(parts[7])
                conf_flag = int(parts[6])
                # MOT 규격: class 1(정상 보행자), conf 1(활성화된 데이터)만 추출
                if class_id == 1 and conf_flag == 1:
                    f_idx = int(parts[0])
                    o_id = int(parts[1])
                    l = float(parts[2])
                    t = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])

                    if f_idx not in gt_data:
                        gt_data[f_idx] = []
                    gt_data[f_idx].append((o_id, l, t, w, h))
    return gt_data


# main.py 내부에 있는 process_mobilenet_frame 함수 수정본
from scipy.optimize import linear_sum_assignment


def process_mobilenet_frame(model_runner: MobileNetSSDTrackerRunner, frame_path: str, imgsz: int, conf_thresh: float):
    frame = cv2.imread(frame_path)
    if frame is None:
        return [], []

    h, w, _ = frame.shape
    resized_frame = cv2.resize(frame, (imgsz, imgsz))
    img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    input_tensor = model_runner.preprocess(input_tensor).unsqueeze(0).to(model_runner.device)

    with torch.no_grad():
        outputs: dict = model_runner.model(input_tensor)[0]

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    scale_w, scale_h = w / imgsz, h / imgsz
    # 💡 힌팅 가두기: high_dets가 딕셔너리를 담은 리스트임을 명시
    high_dets: List[Dict[str, Any]] = []

    for box, score, label in zip(boxes, scores, labels):
        if label == 1 and score >= conf_thresh:
            rescaled_box = [box[0] * scale_w, box[1] * scale_h, box[2] * scale_w, box[3] * scale_h]
            high_dets.append({"box": rescaled_box, "score": float(score)})

    new_tracked: List[SimpleTrack] = []
    matched_det_indices = set()

    if model_runner.tracked_tracks and high_dets:
        tracks_boxes = np.array([t.bbox for t in model_runner.tracked_tracks])
        dets_boxes = np.array([d["box"] for d in high_dets])

        iou_matrix = calculate_iou_matrix(tracks_boxes, dets_boxes)
        cost_matrix = 1.0 - iou_matrix

        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        for t_idx, d_idx in zip(track_indices, det_indices):
            # 💡 핵심 1: 넘파이 인덱스를 int()로 강제 변환하여 시퀀스 소실 방어
            # 💡 핵심 2: 꺼낸 객체가 구조체나 리스트가 아닌 'SimpleTrack' 인스턴스임을 명시
            track: SimpleTrack = model_runner.tracked_tracks[int(t_idx)]
            det: Dict[str, Any] = high_dets[int(d_idx)]

            if iou_matrix[t_idx, d_idx] >= 0.3:
                # 이제 파이참이 det가 dict임을 알고, track이 SimpleTrack임을 완벽히 인지합니다.
                track.bbox = det["box"]
                track.score = det["score"]
                track.lost_frames = 0
                new_tracked.append(track)
                matched_det_indices.add(int(d_idx))
            else:
                track.lost_frames += 1
                if track.lost_frames <= 30:
                    new_tracked.append(track)
    else:
        for track in model_runner.tracked_tracks:
            # 여기서 발생하는 'list에 lost_frames가 없다'는 경고를 타입 고정으로 격파합니다.
            track.lost_frames += 1
            if track.lost_frames <= 30:
                new_tracked.append(track)

    for idx, det in enumerate(high_dets):
        if idx not in matched_det_indices:
            new_track = SimpleTrack(model_runner.next_id, det["box"], det["score"])
            model_runner.next_id += 1
            new_tracked.append(new_track)

    model_runner.tracked_tracks = new_tracked

    pred_ids = []
    pred_boxes = []
    for track in model_runner.tracked_tracks:
        if track.lost_frames == 0:
            pred_ids.append(track.track_id)
            mw = track.bbox[2] - track.bbox[0]
            mh = track.bbox[3] - track.bbox[1]
            pred_boxes.append([track.bbox[0], track.bbox[1], mw, mh])

    return pred_ids, pred_boxes

def main():
    # 1. 설정 매트릭스 동적 로드
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        sys.exit(1)

    scenarios = config.get("scenarios", {})

    # 물리적으로 분할 및 격리 완료된 오리지널 테스트 시퀀스 루트 지정
    test_seq_root = "./data/processed_mot/tracking/test"
    if not os.path.exists(test_seq_root):
        print(f"❌ [CRITICAL] 테스트 데이터 폴더가 없습니다: {test_seq_root}")
        sys.exit(1)

    test_seqs = sorted([os.path.join(test_seq_root, d) for d in os.listdir(test_seq_root)
                        if os.path.isdir(os.path.join(test_seq_root, d))])

    perf_tracker = PerformanceTracker()

    # ======================================================================
    # 🧠 [STAGE 1 & 2 LOG] 최적 가중치 무결성 검증 및 인스턴스 스케줄링
    # ======================================================================
    print("\n" + "=" * 85)
    print("⏳ [STAGE 1 & 2] 아키텍처 챔피언 가중치 보관소 무결성 검증 및 가동 준비")
    print("=" * 85)

    # 자동화 학습 스윕(train.py)을 통해 도출된 베스트 커스텀 가중치 매핑
    yolo_weight = "weights/yolo11_best.pt" if os.path.exists("weights/yolo11_best.pt") else "yolo11n.pt"
    detr_weight = "weights/rtdetr_best.pt" if os.path.exists("weights/rtdetr_best.pt") else "rtdetr-l.pt"

    print(f" ➔ YOLOv11 Target Weight : {yolo_weight}")
    print(f" ➔ RT-DETR Target Weight : {detr_weight}")
    print(f" ➔ MobileNet-SSD         : Torchvision 기성 SSDLite 가속 백본 적용")

    models = {
        "YOLOv11-FT": YOLO(yolo_weight),
        "RT-DETR-FT": RTDETR(detr_weight),
        "MobileNet-SSD": MobileNetSSDTrackerRunner()
    }
    print("✅ [SUCCESS] 3대 정예 패러다임 가속 엔진 로드 완수.")

    # 최종 40라운드 총정리 성적 정산용 저장소
    quantitative_records = []

    # ======================================================================
    # 📊 [STAGE 3 LOG] 시나리오 X 모델 X 다중 영상 교차 벤치마크 가동
    # ======================================================================
    print("\n" + "=" * 85)
    print("🚀 [STAGE 3 : TEST LOG] 가변 파라미터 5대 시나리오 정량 평가 세션 돌입")
    print("=" * 85)

    for scenario_name, params in scenarios.items():
        print(f"\n🎬 [SCENARIO RUN] ➔ {scenario_name} ({params.get('description', '')})")
        print("-" * 85)

        for model_name, model_engine in models.items():
            print(f" 🤖 가동 모델 패러다임: {model_name}")

            # 하드웨어 변동 인자 추출 파싱 및 강제 형변환 (PyCharm 타입 경고 완벽 제거)
            if model_name in ["YOLOv11-FT", "RT-DETR-FT"]:
                args = params.get("ultralytics_args", {})
            else:
                args = params.get("mobilenet_args", {})

            imgsz_arg = int(args.get("imgsz", 640))
            conf_arg = float(args.get("conf", 0.25))
            track_buffer_arg = int(args.get("track_buffer", 30))

            # 격리된 다중 영상 세트 순회 시작
            for seq_path in test_seqs:
                seq_name = os.path.basename(seq_path)
                print(f"   📺 [VIDEO LEVEL] Target Sequence: {seq_name}")

                # 정답 정보 스트림 파싱 적재
                gt_frames = parse_gt_for_tracking(seq_path)

                # 글로벌 표준 평가지표 계산용 motmetrics 누산기 셋업
                accumulator = mm.MOTAccumulator(auto_id=True)

                img_dir = os.path.join(seq_path, "img1")
                frame_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg'))])

                # 하드웨어 타이머 및 VRAM 레지스터 초기화
                perf_tracker.start_session()

                # ------------------------------------------------------------------
                # ⚡ [FRAME LEVEL LOOP LOG] 프레임 단위 실시간 연산 소모 및 추적 정산
                # ------------------------------------------------------------------
                # PyCharm 터미널 가독성을 극대화하는 tqdm 프로그레스 바 바인딩
                for frame_file in tqdm(frame_files, desc=f"    ✨ Frame Tracking Loop", leave=False):
                    f_idx = int(os.path.splitext(frame_file)[0])
                    frame_full_path = os.path.join(img_dir, frame_file)

                    pred_ids = []
                    pred_boxes = []

                    # 💡 변경: 문자열 검사 대신, 실제 클래스 인스턴스 검사(isinstance)를 수행합니다.
                    # 이렇게 하면 if문 안쪽 블록에서는 model_engine이 무조건 track을 가진 모델로 좁혀집니다(Narrowing).
                    if isinstance(model_engine, (YOLO, RTDETR)):
                        res = model_engine.track(
                            source=frame_full_path,
                            imgsz=imgsz_arg,
                            conf=conf_arg,
                            persist=True,
                            verbose=False
                        )[0]

                        if res.boxes.id is not None:
                            boxes = res.boxes.xywh.cpu().numpy()
                            ids = res.boxes.id.cpu().numpy().astype(int)
                            clss = res.boxes.cls.cpu().numpy()

                            for box, o_id, cls in zip(boxes, ids, clss):
                                if int(cls) == 0:
                                    l = float(box[0] - (box[2] / 2.0))
                                    t = float(box[1] - (box[3] / 2.0))
                                    pred_ids.append(int(o_id))
                                    pred_boxes.append([l, t, float(box[2]), float(box[3])])

                    elif isinstance(model_engine, MobileNetSSDTrackerRunner):
                        # MobileNet일 때는 커스텀 헝가리안 브릿지 매칭 호출
                        pred_ids, pred_boxes = process_mobilenet_frame(
                            model_engine, frame_full_path, imgsz_arg, conf_arg
                        )

                    # 프레임 정산 카운트 및 실시간 최대 메모리 갱신
                    perf_tracker.count_frame()
                    perf_tracker.update_vram()

                    # 현재 프레임 정답 데이터 매핑 준비
                    gt_data = gt_frames.get(f_idx, [])
                    gt_ids = [g[0] for g in gt_data]
                    gt_boxes = [[g[1], g[2], g[3], g[4]] for g in gt_data]

                    # 교차 IoU 거리 행렬 연산 후 누산기 업데이트
                    distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
                    accumulator.update(gt_ids, pred_ids, distances)

                # ------------------------------------------------------------------
                # 🎬 [VIDEO LEVEL LOG] 한 개 영상 시퀀스 종료 시점 정량 스코어 정산
                # ------------------------------------------------------------------
                hardware_metrics = perf_tracker.end_session()
                mh = mm.metrics.create()
                summary = mh.compute(
                    accumulator,
                    metrics=['mota', 'idf1', 'num_swaps', 'num_misses', 'num_false_positives'],
                    name=seq_name
                )

                # 데이터 레이아웃 추출 (% 스케일링)
                mota_val = float(summary['mota'].iloc[0] * 100)
                idf1_val = float(summary['idf1'].iloc[0] * 100)
                swaps = int(summary['num_swaps'].iloc[0])
                fp = int(summary['num_false_positives'].iloc[0])
                fn = int(summary['num_misses'].iloc[0])

                print(
                    f"    ➔ [RESULT] FPS: {hardware_metrics['fps']:.2f} | VRAM: {hardware_metrics['vram_gb']:.2f}GB | MOTA: {mota_val:.1f}% | ID Swaps: {swaps} | FP: {fp} | FN: {fn}")

                # 최종 CSV 보존용 데이터 적재
                quantitative_records.append({
                    "Scenario": scenario_name,
                    "Model": model_name,
                    "Sequence": seq_name,
                    "FPS": hardware_metrics['fps'],
                    "Peak_VRAM_GB": hardware_metrics['vram_gb'],
                    "MOTA_Percent": round(mota_val, 2),
                    "IDF1_Percent": round(idf1_val, 2),
                    "ID_Swaps": swaps,
                    "False_Positives_FP": fp,
                    "False_Negatives_FN": fn,
                    "Total_Frames": hardware_metrics['total_frames'],
                    "Total_Time_Sec": hardware_metrics['total_time_sec']
                })

    # ======================================================================
    # 💾 [FINAL SUMMARY LOG] 벤치마크 총정리 성적표 마스터 테이블 출력 및 파일 저장
    # ======================================================================
    print("\n" + "=" * 85)
    print("📊 [종합 정량 리포트] 하이드웨어 및 트래킹 가변 변동 지표 정산 스코어보드")
    print("=" * 85)

    df_final = pd.DataFrame(quantitative_records)

    # results 폴더 부재 시 자동 생성 후 결과 백업
    os.makedirs("./results", exist_ok=True)
    output_path = "./results/complete_benchmark_report.csv"
    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 판다스 출력 포맷 설정 최적화로 깔끔한 콘솔 표 구현
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_final[["Scenario", "Model", "Sequence", "FPS", "Peak_VRAM_GB", "MOTA_Percent", "ID_Swaps",
                    "False_Positives_FP"]].to_string(index=False))

    print("\n" + "=" * 85)
    print(f"🎉 전 과정 성능 실험 완수! 최종 보고서용 데이터 드랍 완료 ➔ '{output_path}'")
    print("=" * 85)


if __name__ == "__main__":
    main()