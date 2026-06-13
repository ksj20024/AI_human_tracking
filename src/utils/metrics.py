import time
import torch

class PerformanceTracker:
    def __init__(self):
        self.start_time = None
        self.frame_count = 0
        self.peak_vram = 0

    def start_session(self):
        """실험 라운드 시작 시 타이머와 GPU 메모리를 초기화합니다."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self.start_time = time.time()
        self.frame_count = 0
        self.peak_vram = 0

    def count_frame(self):
        """프레임이 처리될 때마다 호출하여 카운트를 올립니다."""
        self.frame_count += 1

    def update_vram(self):
        """현재 라운드의 최대 GPU 메모리 점유율(VRAM)을 업데이트합니다."""
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            self.peak_vram = peak_bytes / (1024 ** 3)  # Bytes -> GB
        else:
            self.peak_vram = 0.0

    def end_session(self):
        """실험 결과를 정산하여 FPS와 VRAM 값을 반환합니다."""
        if self.start_time is None:
            return {"fps": 0, "vram_gb": 0}

        end_time = time.time()
        total_time = end_time - self.start_time
        fps = self.frame_count / total_time if total_time > 0 else 0
        self.update_vram()

        return {
            "fps": round(fps, 2),
            "vram_gb": round(self.peak_vram, 3),
            "total_time_sec": round(total_time, 2),
            "total_frames": self.frame_count
        }