# train.py
import os
import shutil
import pandas as pd
from datetime import datetime
from ultralytics import YOLO, RTDETR


def run_automated_tuning_sweep():
    # 초기 환경 구축
    yaml_path = "./data/mot_transfer.yaml"
    output_weights_dir = "./weights"
    os.makedirs(output_weights_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("TRAIN LOG : 하이퍼파라미터 전이학습 시작")
    print("=" * 80)
    print(f"START TIME : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TARGET DATA : {yaml_path}")

    # 런타임 자원, 속도를 고려해 Learning Rate와 Batch Size를 교차 검증
    yolo_tuning_grid = [
        {"lr0": 0.01, "batch": 16, "epochs": 50, "optimizer": "SGD"},
        {"lr0": 0.001, "batch": 16, "epochs": 50, "optimizer": "AdamW"},
        {"lr0": 0.005, "batch": 16, "epochs": 50, "optimizer": "SGD"},
        {"lr0": 0.005, "batch": 16, "epochs": 50, "optimizer": "AdamW"},
        #{"lr0": 0.01, "batch": 32, "epochs": 50, "optimizer": "SGD"},  # 메모리 문제로 g4dn.xlarge에서 돌아가지 않음
        {"lr0": 0.01, "batch": 8, "epochs": 50, "optimizer": "SGD"},
    ]

    # 모델 크기가 크므로 에포크를 낮춰서(15) 경향성만 파악
    detr_tuning_grid = [
        {"lr0": 0.01, "batch": 8, "epochs": 15, "optimizer": "SGD"},
        {"lr0": 0.001, "batch": 8, "epochs": 15, "optimizer": "AdamW"}
    ]

    tuning_records = []

    # ======================================================================
    # YOLOv11 파인튜닝
    # ======================================================================
    print("\n" + "-" * 50)
    print(" MODEL : YOLOv11 (CNN 패러다임) 파인튜닝 시작")
    print("-" * 50)

    best_yolo_map = -1.0

    for run_idx, params in enumerate(yolo_tuning_grid):
        print(f"\n YOLOv11 Run : {run_idx + 1}/{len(yolo_tuning_grid)} Parameters 주입:")
        print(
            f" ➔ LR: {params['lr0']} | Batch: {params['batch']} | Epochs: {params['epochs']} | Opt: {params['optimizer']}")

        # 기성 가중치 로드
        model = YOLO("yolo11n.pt")

        # 파인튜닝 가동 (실시간 에포크 로그는 내부 프레임워크가 콘솔에 트리거)
        results = model.train(
            data=yaml_path,
            epochs=params["epochs"],
            batch=params["batch"],
            lr0=params["lr0"],
            optimizer=params["optimizer"],
            device=0,  # GPU 0번 강제 할당
            project="runs_tuning_yolo",
            name=f"yolo_sweep_{run_idx + 1}",
            verbose=True,  # 각 에포크별 Loss, Precision, Recall 정량 로그 표출
            workers = 2
        )

        # 학습 종료 후 최종 에포크 검증 성적 스코어 파싱
        if results is not None:
            final_map50 = results.results_dict["metrics/mAP50(B)"]
            train_loss = results.results_dict.get("val/box_loss", 0.0)
        else:
            final_map50 = 0.0
            train_loss = 0.0

        print(f" TRAINNING FINISH : 검증 성적 정산 -> mAP50: {final_map50:.4f} | Box Loss: {train_loss:.4f}")

        tuning_records.append({
            "Architecture": "YOLOv11", "Run_ID": run_idx + 1, "LR": params["lr0"],
            "Batch": params["batch"], "Optimizer": params["optimizer"],
            "mAP50": round(final_map50, 4), "Val_Loss": round(train_loss, 4)
        })

        # 최고 모델 판별 및 가중치 백업
        if final_map50 > best_yolo_map and results is not None:
            best_yolo_map = final_map50
            best_weight_src = os.path.join(results.save_dir, "weights", "best.pt")
            best_weight_dst = os.path.join(output_weights_dir, "yolo11_best.pt")
            shutil.copy(best_weight_src, best_weight_dst)
            print(f" BEST MODEL UPDATED : 최고 mAP 가중치 복사 완료 ➔ {best_weight_dst}")

    # ======================================================================
    # RT-DETR 파인 튜닝
    # ======================================================================
    print("\n" + "-" * 50)
    print(" MODEL : RT-DETR (Transformer 패러다임) 파인튜닝 시작")
    print("-" * 50)

    best_detr_map = -1.0

    for run_idx, params in enumerate(detr_tuning_grid):
        print(f"\n RT-DETR Run : {run_idx + 1}/{len(detr_tuning_grid)} Parameters 주입:")
        # Transformer 계열은 VRAM 부하가 극심하므로 안전 가이드라인 적용하여 배치 크기 절반 하향 조절
        safe_batch = max(2, int(params["batch"]) // 2)
        print(
            f" ➔ LR: {params['lr0']} | Safe Batch: {safe_batch} | Epochs: {params['epochs']} | Opt: {params['optimizer']}")

        model = RTDETR("rtdetr-l.pt")

        results = model.train(
            data=yaml_path,
            epochs=params["epochs"],
            batch=safe_batch,
            lr0=params["lr0"],
            optimizer=params["optimizer"],
            device=0,
            project="runs_tuning_detr",
            name=f"detr_sweep_{run_idx + 1}",
            verbose=True,
            workers = 4
        )

        if results is not None:
            final_map50 = results.results_dict["metrics/mAP50(B)"]
            train_loss = results.results_dict.get("val/slice_loss", 0.0)
        else:
            final_map50 = 0.0
            train_loss = 0.0

        print(f" RUN FINISHED LOG : 검증 결과 -> mAP50: {final_map50:.4f}")

        tuning_records.append({
            "Architecture": "RT-DETR", "Run_ID": run_idx + 1, "LR": params["lr0"],
            "Batch": safe_batch, "Optimizer": params["optimizer"],
            "mAP50": round(final_map50, 4), "Val_Loss": round(train_loss, 4)
        })

        if final_map50 > best_detr_map and results is not None:
            best_detr_map = final_map50
            best_weight_src = os.path.join(results.save_dir, "weights", "best.pt")
            best_weight_dst = os.path.join(output_weights_dir, "rtdetr_best.pt")
            shutil.copy(best_weight_src, best_weight_dst)
            print(f" BEST MODEL UPDATED : 최고 mAP 가중치 복사 완료 ➔ {best_weight_dst}")

    # ======================================================================
    # 하이퍼파라미터 최종 결과 출력 및 저장
    # ======================================================================
    print("\n" + "=" * 80)
    print(" FINAL REPORT : 하이퍼파라미터 측정 최종 결과")
    print("=" * 80)

    df_sweep = pd.DataFrame(tuning_records)
    df_sweep.to_csv(os.path.join(output_weights_dir, "hyperparameter_log_report.csv"), index=False,
                    encoding="utf-8-sig")
    print(df_sweep.to_string(index=False))

    print(f"\n 학습 및 최고 가중치 측정 종료. 리포트 저장 완료.")
    print("=" * 80)


if __name__ == "__main__":
    run_automated_tuning_sweep()