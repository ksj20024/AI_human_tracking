# src/models/mobilenet_tracker.py
import torch
import numpy as np
from typing import List

from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights


class SimpleTrack:
    def __init__(self, t_id: int, bbox: list, score: float):
        self.track_id = t_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.lost_frames = 0


def calculate_iou_matrix(tracks_boxes, dets_boxes):
    if len(tracks_boxes) == 0 or len(dets_boxes) == 0:
        return np.zeros((len(tracks_boxes), len(dets_boxes)))
    b1 = tracks_boxes[:, np.newaxis, :]
    b2 = dets_boxes[np.newaxis, :, :]
    x_a = np.maximum(b1[..., 0], b2[..., 0])
    y_a = np.maximum(b1[..., 1], b2[..., 1])
    x_b = np.minimum(b1[..., 2], b2[..., 2])
    y_b = np.minimum(b1[..., 3], b2[..., 3])
    inter_area = np.maximum(0.0, x_b - x_a) * np.maximum(0.0, y_b - y_a)
    box_a_area = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    box_b_area = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union_area = box_a_area + box_b_area - inter_area
    return np.where(union_area > 0, inter_area / union_area, 0.0)


class MobileNetSSDTrackerRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights).to(self.device)
        self.model.eval()

        self.next_id = 1
        # 💡 핵심: 내부 리스트가 SimpleTrack 전용 리스트임을 파이참에게 명시
        self.tracked_tracks: List[SimpleTrack] = []
        self.lost_tracks: List[SimpleTrack] = []