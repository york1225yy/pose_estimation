"""Video clip dataset for action recognition.

Each sample is a short video clip specified by (video_path, frame_start, frame_end)
from the activity label CSV.  Frames are extracted with decord, resized and
normalised to match ImageNet statistics, then returned as a tensor ready for
both TransDARC and UniFormerV2.

Tensor layout: (C=3, T, H, W)  — channels-first, T frames
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from decord import VideoReader, cpu
    _DECORD = True
except ImportError:
    import cv2
    _DECORD = False


# ImageNet mean / std (used by both ViT-based models)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize shortest edge to `size` with aspect-ratio preservation."""
    import cv2 as _cv2
    h, w = frame.shape[:2]
    if h <= w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    return _cv2.resize(frame, (new_w, new_h), interpolation=_cv2.INTER_LINEAR)


def _center_crop(frame: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return frame[y0:y0 + crop_size, x0:x0 + crop_size]


def _random_crop(frame: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    y0 = np.random.randint(0, max(1, h - crop_size))
    x0 = np.random.randint(0, max(1, w - crop_size))
    return frame[y0:y0 + crop_size, x0:x0 + crop_size]


def _read_frames_decord(video_path: str, frame_indices: np.ndarray) -> np.ndarray:
    """Return (T, H, W, 3) uint8 array using decord."""
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, 3) RGB
    return frames


def _read_frames_cv2(video_path: str, frame_indices: np.ndarray) -> np.ndarray:
    """Fallback: read frames with OpenCV."""
    import cv2 as _cv2
    cap = _cv2.VideoCapture(video_path)
    frames = []
    prev = -1
    for idx in sorted(frame_indices):
        if idx != prev:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
        else:
            # Pad with zeros if frame unreadable
            if frames:
                frames.append(np.zeros_like(frames[-1]))
            else:
                frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
        prev = idx + 1
    cap.release()
    # Re-order to original frame_indices order
    sort_order = np.argsort(frame_indices)
    result = [None] * len(frame_indices)
    for pos, orig_pos in enumerate(sort_order):
        result[orig_pos] = frames[pos]
    return np.stack(result)


def _sample_frame_indices(frame_start: int, frame_end: int,
                          num_frames: int, temporal_stride: int) -> np.ndarray:
    """
    Sample `num_frames` indices from [frame_start, frame_end).
    Uses uniform stride, then clips to valid range.
    """
    length = max(frame_end - frame_start, 1)
    step = max(length // num_frames, 1)
    indices = np.arange(frame_start, frame_start + step * num_frames, step)[:num_frames]
    # Wrap any overshoot back to last valid frame
    indices = np.clip(indices, frame_start, frame_end - 1)
    return indices.astype(np.int64)


def _preprocess(frames: np.ndarray, crop_size: int, augment: bool) -> torch.Tensor:
    """
    frames: (T, H, W, 3) uint8 RGB
    Returns: (3, T, crop_size, crop_size) float32 tensor, normalised
    """
    resize_size = int(crop_size * 256 / 224)  # ~292 for crop 256, 256 for crop 224
    out = []
    for i, frame in enumerate(frames):
        frame = _resize_frame(frame, resize_size)
        if augment:
            if i == 0:
                # Decide once per clip whether to flip
                _do_flip = (np.random.rand() < 0.5)
                _do_jitter = (np.random.rand() < 0.5)
                _brightness = np.random.uniform(0.8, 1.2) if _do_jitter else 1.0
            frame = _random_crop(frame, crop_size)
            if _do_flip:
                frame = frame[:, ::-1, :].copy()
            frame = np.clip(frame.astype(np.float32) * _brightness, 0, 255).astype(np.uint8)
        else:
            frame = _center_crop(frame, crop_size)

        frame = frame.astype(np.float32) / 255.0
        frame = (frame - _MEAN) / _STD
        out.append(frame)

    # (T, H, W, 3) -> (3, T, H, W)
    arr = np.stack(out, axis=0).transpose(3, 0, 1, 2).astype(np.float32)
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# File-ID → video path mapping
# ---------------------------------------------------------------------------

def _build_video_mapping(video_dir: str, file_ids):
    """
    Map file_ids like 'vp1/run1b_2018-05-29-14-02-47.ids_1' to MP4 paths.
    Only vp1 file_ids have corresponding videos in video_dir.
    """
    mapping = {}
    suffix = '.mp4'
    available = {
        fname[:-len(suffix)]: os.path.join(video_dir, fname)
        for fname in os.listdir(video_dir)
        if fname.endswith(suffix)
    }  # {stem: full_path}

    for fid in file_ids:
        base = fid.split('/', 1)[-1]   # strip 'vpX/' prefix
        if base in available:
            mapping[fid] = available[base]
    return mapping


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoDataset(Dataset):
    """
    Video clip dataset aligned with the Drive&Act label CSVs.

    Each sample: (clip_tensor, label)
        clip_tensor : (3, T, crop_size, crop_size) float32
        label       : int class index 0..num_classes-1

    Args:
        label_csv       : Path to activity label CSV
        video_dir       : Directory containing *.mp4 files
        allowed_classes : Optional list of class names to keep (remapped 0..N-1)
        num_frames      : Number of frames to sample per clip (default: 8)
        crop_size       : Spatial crop size in pixels (default: 224)
        augment         : Whether to apply spatial augmentation
    """

    def __init__(self, label_csv: str, video_dir: str,
                 allowed_classes=None, num_frames: int = 8,
                 crop_size: int = 224, augment: bool = False):
        self.video_dir  = video_dir
        self.num_frames = num_frames
        self.crop_size  = crop_size
        self.augment    = augment

        labels_df = pd.read_csv(label_csv)
        all_file_ids = labels_df['file_id'].unique().tolist()
        video_map = _build_video_mapping(video_dir, all_file_ids)

        # Keep only rows with an available video
        labels_df = labels_df[labels_df['file_id'].isin(video_map)].reset_index(drop=True)

        # Filter classes
        if allowed_classes is not None:
            labels_df = labels_df[labels_df['activity'].isin(allowed_classes)].reset_index(drop=True)
            self.activity_labels = list(allowed_classes)
        else:
            self.activity_labels = sorted(labels_df['activity'].unique().tolist())

        self.activity_to_idx = {name: idx for idx, name in enumerate(self.activity_labels)}
        self.num_classes = len(self.activity_labels)

        # Drop rows whose activity is not in our label set (safety)
        labels_df = labels_df[labels_df['activity'].isin(self.activity_to_idx)].reset_index(drop=True)
        self.samples  = labels_df
        self.video_map = video_map

        print(f"VideoDataset: {len(video_map)} video(s) found, "
              f"{len(self.samples)} clip rows matched")
        print(f"Classes ({self.num_classes}): {self.activity_labels}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row         = self.samples.iloc[idx]
        file_id     = row['file_id']
        frame_start = int(row['frame_start'])
        frame_end   = int(row['frame_end'])
        activity    = row['activity']
        label       = self.activity_to_idx[activity]

        video_path = self.video_map[file_id]
        frame_indices = _sample_frame_indices(frame_start, frame_end, self.num_frames, 1)

        if _DECORD:
            frames = _read_frames_decord(video_path, frame_indices)
        else:
            frames = _read_frames_cv2(video_path, frame_indices)

        clip = _preprocess(frames, self.crop_size, self.augment)
        return clip, label
