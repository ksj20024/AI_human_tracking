import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd

# 🚨 [NumPy 2.0+ 호환성 긴급 패치]
# motmetrics 내부에서 삭제된 np.asfarray를 호출해 터지는 억까를 원천 차단합니다.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, *args, **kwargs: np.asarray(a, dtype=float, *args, **kwargs)

import motmetrics as mm
from tqdm import tqdm
from ultralytics import YOLO, RTDETR
from typing import List, Dict, Any
from scipy.optimize import linear_sum_assignment
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
            track: SimpleTrack = model_runner.tracked_tracks[int(t_idx)]
            det: Dict[str, Any] = high_dets[int(d_idx)]

            if iou_matrix[t_idx, d_idx] >= 0.3:
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
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        sys.exit(1)

    scenarios = config.get("scenarios", {})

    test_seq_root = "./data/processed_mot/tracking/test"
    if not os.path.exists(test_seq_root):
        print(f"❌ [CRITICAL] 테스트 데이터 폴더가 없습니다: {test_seq_root}")
        sys.exit(1)

    test_seqs = sorted([os.path.join(test_seq_root, d) for d in os.listdir(test_seq_root)
                        if os.path.isdir(os.path.join(test_seq_root, d))])

    perf_tracker = PerformanceTracker()

    print("\n" + "=" * 85)
    print("⏳ [STAGE 1 & 2] 순정(Pretrained) vs 파인튜닝(FT) 아키텍처 다중 로드 세션 가동")
    print("=" * 85)

    # 💡 [핵심 변경 1] 순정 모델과 파인튜닝 모델 독립 선언 및 가중치 경로 구체화
    models = {
        "YOLOv11-Pure": YOLO("yolo11n.pt"),
        "RT-DETR-Pure": RTDETR("rtdetr-l.pt"),
        "MobileNet-SSD": MobileNetSSDTrackerRunner()
    }

    # YOLO 파인튜닝 가중치 자동 탐색 및 조건부 추가
    yolo_ft_candidates = [
        "weights/yolo11_best.pt",
        "runs_tuning_yolo/yolo_sweep_fixed/weights/best.pt"
    ]
    for path in yolo_ft_candidates:
        if os.path.exists(path):
            models["YOLOv11-FT"] = YOLO(path)
            print(f" LOADED : YOLOv11 Fine-Tuned 모델 바인딩 성공 ➔ {path}")
            break

    # RT-DETR 파인튜닝 가중치 자동 탐색 및 조건부 추가 (last.pt 강종 대비용 포함)
    detr_ft_candidates = [
        "weights/rtdetr_best.pt",
        "runs_tuning_detr/detr_sweep_1/weights/best.pt",
        "runs_tuning_detr/detr_sweep_1/weights/last.pt"  # 👈 도중 중단 가중치 구제책
    ]
    for path in detr_ft_candidates:
        if os.path.exists(path):
            models["RT-DETR-FT"] = RTDETR(path)
            print(f" 🐘 [LOADED] RT-DETR Fine-Tuned 모델 바인딩 성공 ➔ {path}")
            break

    print(f" ➔ 활성화된 총 모델 개수: {len(models)}개")
    print("SUCCESS : 벤치마크 대상 모델 선택 완료.")

    quantitative_records = []

    print("\n" + "=" * 85)
    print(" test 시작 : 가변 파라미터 시나리오 평가 시작")
    print("=" * 85)

    for scenario_name, params in scenarios.items():
        print(f"\n🎬 [SCENARIO RUN] ➔ {scenario_name} ({params.get('description', '')})")
        print("-" * 85)

        for model_name, model_engine in models.items():
            print(f" 가동 모델 : {model_name}")

            # 💡 [핵심 변경 2] 하드웨어 변동 인자 추출 파싱 로직 확장 (Pure 모델 이름도 수용하도록 수정)
            if "YOLO" in model_name or "DETR" in model_name:
                args = params.get("ultralytics_args", {})
            else:
                args = params.get("mobilenet_args", {})

            imgsz_arg = int(args.get("imgsz", 640))
            conf_arg = float(args.get("conf", 0.25))
            track_buffer_arg = int(args.get("track_buffer", 30))

            for seq_path in test_seqs:
                seq_name = os.path.basename(seq_path)
                print(f" VIDEO Target Sequence: {seq_name}")

                gt_frames = parse_gt_for_tracking(seq_path)
                accumulator = mm.MOTAccumulator(auto_id=True)

                img_dir = os.path.join(seq_path, "img1")
                frame_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg'))])

                perf_tracker.start_session()

                for frame_file in tqdm(frame_files, desc=f"    ✨ Frame Tracking Loop", leave=False):
                    f_idx = int(os.path.splitext(frame_file)[0])
                    frame_full_path = os.path.join(img_dir, frame_file)

                    pred_ids = []
                    pred_boxes = []

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
                                if int(cls) == 0:  # COCO 규격 0번 (Person)
                                    l = float(box[0] - (box[2] / 2.0))
                                    t = float(box[1] - (box[3] / 2.0))
                                    pred_ids.append(int(o_id))
                                    pred_boxes.append([l, t, float(box[2]), float(box[3])])

                    elif isinstance(model_engine, MobileNetSSDTrackerRunner):
                        pred_ids, pred_boxes = process_mobilenet_frame(
                            model_engine, frame_full_path, imgsz_arg, conf_arg
                        )

                    perf_tracker.count_frame()
                    perf_tracker.update_vram()

                    gt_data = gt_frames.get(f_idx, [])
                    gt_ids = [g[0] for g in gt_data]
                    gt_boxes = [[g[1], g[2], g[3], g[4]] for g in gt_data]

                    distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
                    accumulator.update(gt_ids, pred_ids, distances)

                hardware_metrics = perf_tracker.end_session()
                mh = mm.metrics.create()
                summary = mh.compute(
                    accumulator,
                    metrics=['mota', 'idf1', 'num_switches', 'num_misses', 'num_false_positives'],
                    name=seq_name
                )

                mota_val = float(summary['mota'].iloc[0] * 100)
                idf1_val = float(summary['idf1'].iloc[0] * 100)
                swaps = int(summary['num_switches'].iloc[0])
                fp = int(summary['num_false_positives'].iloc[0])
                fn = int(summary['num_misses'].iloc[0])

                print(
                    f"    ➔ [RESULT] FPS: {hardware_metrics['fps']:.2f} | VRAM: {hardware_metrics['vram_gb']:.2f}GB | MOTA: {mota_val:.1f}% | ID Swaps: {swaps} | FP: {fp} | FN: {fn}")

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

    print("\n" + "=" * 85)
    print("종합 정량 리포트 하드웨어 및 트래킹 가변 변동 지표 정산 스코어보드")
    print("=" * 85)

    df_final = pd.DataFrame(quantitative_records)
    os.makedirs("./results", exist_ok=True)
    output_path = "./results/complete_benchmark_report.csv"
    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_final[["Scenario", "Model", "Sequence", "FPS", "Peak_VRAM_GB", "MOTA_Percent", "ID_Swaps",
                    "False_Positives_FP"]].to_string(index=False))

    print("\n" + "=" * 85)
    print(f" 전 과정 성능 실험 완료. 최종 보고서용 데이터 출력 완료 ➔ '{output_path}'")
    print("=" * 85)


if __name__ == "__main__":
    main()