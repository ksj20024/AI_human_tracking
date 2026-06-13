import os
import yaml
from ultralytics import YOLO, RTDETR

class UltralyticsTrackerRunner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """weights/ 폴더 구조에 매핑되도록 가중치 로드 경로를 지정합니다."""
        os.makedirs("weights", exist_ok=True)
        if self.model_name == "YOLOv11":
            self.model = YOLO("weights/yolo11n.pt")
        elif self.model_name == "RT-DETR":
            self.model = RTDETR("weights/rtdetr-l.pt")
        else:
            raise ValueError(f"지원하지 않는 모델입니다: {self.model_name}")

    def _create_temporary_tracker_config(self, track_buffer: int) -> str:
        """가변적인 track_buffer를 적용한 임시 ByteTrack YAML 파일을 생성합니다."""
        config_dict = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.25,
            "track_buffer": track_buffer,
            "match_thresh": 0.8,
            "fuse_score": True
        }
        temp_yaml_path = f"config/temp_bytetrack_{track_buffer}.yaml"
        os.makedirs("config", exist_ok=True)
        with open(temp_yaml_path, "w") as f:
            yaml.dump(config_dict, f)
        return temp_yaml_path

    def run_inference(self, video_path: str, args: dict, perf_tracker) -> dict:
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        imgsz = args.get("imgsz", 640)
        conf = args.get("conf", 0.25)
        track_buffer = args.get("track_buffer", 30)

        tracker_yaml = self._create_temporary_tracker_config(track_buffer)
        perf_tracker.start_session()

        try:
            results = self.model.track(
                source=video_path,
                imgsz=imgsz,
                conf=conf,
                tracker=tracker_yaml,
                stream=True,
                persist=True,
                verbose=False
            )
            for _ in results:
                perf_tracker.count_frame()
                perf_tracker.update_vram()
        finally:
            if os.path.exists(tracker_yaml):
                os.remove(tracker_yaml)

        return perf_tracker.end_session()