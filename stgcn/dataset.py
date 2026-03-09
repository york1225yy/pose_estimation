"""Dataset for loading skeleton pose data with activity labels."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .graph import JOINT_NAMES


# Activity label mapping
ACTIVITY_LABELS = [
    'fasten_seat_belt', 'hand_over', 'work', 'eat_drink',
    'read_write_newspaper', 'read_write_magazine', 'watch_video',
    'put_on_jacket', 'take_off_jacket', 'put_on_sunglasses',
    'take_off_sunglasses', 'final_task',
]
ACTIVITY_TO_IDX = {name: idx for idx, name in enumerate(ACTIVITY_LABELS)}
NUM_CLASSES = len(ACTIVITY_LABELS)


def _build_file_id_mapping(pose_dir, file_ids):
    """
    Dynamically build {file_id: csv_path} for all available pose CSVs.

    Each pose CSV is named like:  run1_2018-05-03-14-08-31.ids_1.openpose.3d.csv
    Each label file_id is like:   vp2/run1_2018-05-03-14-08-31.ids_1

    The match is: file_id ends with '/' + csv_stem  (csv_stem strips '.openpose.3d.csv')
    """
    mapping = {}
    suffix = '.openpose.3d.csv'
    available = {
        fname[:-len(suffix)]: os.path.join(pose_dir, fname)
        for fname in os.listdir(pose_dir)
        if fname.endswith(suffix)
    }  # {stem: full_path}

    for fid in file_ids:
        # fid example: "vp2/run1_2018-05-03-14-08-31.ids_1"
        base = fid.split('/', 1)[-1]  # strip vpX/ prefix
        if base in available:
            mapping[fid] = available[base]
    return mapping


class PoseDataset(Dataset):
    """
    Skeleton action recognition dataset.

    Each sample: (skeleton_tensor, label)
        skeleton_tensor: (C=3, T, V=25) float32 — x, y, z for each joint per frame
        label: int class index
    """

    def __init__(self, label_csv, pose_dir, max_frames=90, augment=False):
        """
        Args:
            label_csv: path to activity label CSV (e.g., split_0.train.csv)
            pose_dir: path to directory containing openpose .3d.csv files
            max_frames: temporal window size (pad/crop to this length)
            augment: whether to apply data augmentation
        """
        self.pose_dir = pose_dir
        self.max_frames = max_frames
        self.augment = augment

        # Load all labels, then keep only rows whose pose CSV is available
        labels_df = pd.read_csv(label_csv)
        all_file_ids = labels_df['file_id'].unique().tolist()
        file_id_map = _build_file_id_mapping(pose_dir, all_file_ids)

        labels_df = labels_df[labels_df['file_id'].isin(file_id_map)].reset_index(drop=True)
        self.samples = labels_df

        # Pre-load all available pose CSVs into memory (keyed by file_id)
        self.pose_data = {}
        for file_id, csv_path in file_id_map.items():
            df = pd.read_csv(csv_path)
            self.pose_data[file_id] = df

        if not self.pose_data:
            raise RuntimeError(
                f"No matching pose CSVs found in '{pose_dir}' for the given label file. "
                "Check that pose_dir contains *.openpose.3d.csv files."
            )

        available_ids = set(self.pose_data.keys())
        matched = labels_df['file_id'].isin(available_ids).sum()
        print(f"Pose data: {len(self.pose_data)} CSV file(s) loaded, "
              f"{matched} label rows matched ({len(self.samples)} samples total)")

        # Build column mapping: for each joint, find x, y, z column indices
        sample_df = next(iter(self.pose_data.values()))
        cols = list(sample_df.columns)
        self.joint_col_indices = []  # List of (x_idx, y_idx, z_idx) for each joint
        for joint_name in JOINT_NAMES:
            x_col = f'{joint_name}_x'
            y_col = f'{joint_name}_y'
            z_col = f'{joint_name}_z'
            self.joint_col_indices.append((
                cols.index(x_col), cols.index(y_col), cols.index(z_col)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        file_id = row['file_id']
        frame_start = int(row['frame_start'])
        frame_end = int(row['frame_end'])
        activity = row['activity']
        label = ACTIVITY_TO_IDX[activity]

        # Extract skeleton frames
        df = self.pose_data[file_id]
        num_total = len(df)

        # Clip to valid range
        frame_start = max(0, min(frame_start, num_total - 1))
        frame_end = max(frame_start + 1, min(frame_end, num_total))

        segment = df.iloc[frame_start:frame_end]
        values = segment.values  # numpy array (T_actual, num_cols)

        # Extract joint coordinates: shape (T_actual, V, 3)
        T_actual = values.shape[0]
        V = len(JOINT_NAMES)
        skeleton = np.zeros((T_actual, V, 3), dtype=np.float32)
        for j, (xi, yi, zi) in enumerate(self.joint_col_indices):
            skeleton[:, j, 0] = values[:, xi].astype(np.float32)
            skeleton[:, j, 1] = values[:, yi].astype(np.float32)
            skeleton[:, j, 2] = values[:, zi].astype(np.float32)

        # Pad or crop to max_frames
        if T_actual < self.max_frames:
            pad = np.zeros((self.max_frames - T_actual, V, 3), dtype=np.float32)
            skeleton = np.concatenate([skeleton, pad], axis=0)
        elif T_actual > self.max_frames:
            skeleton = skeleton[:self.max_frames]

        # Data augmentation
        if self.augment:
            skeleton = self._augment(skeleton)

        # Convert to (C=3, T, V)
        skeleton = skeleton.transpose(2, 0, 1)  # (3, T, V)
        return torch.from_numpy(skeleton), label

    def _augment(self, skeleton):
        """Simple augmentation: random noise, random scale, temporal shift."""
        # Random Gaussian noise
        if np.random.rand() < 0.5:
            noise = np.random.randn(*skeleton.shape).astype(np.float32) * 0.01
            skeleton = skeleton + noise

        # Random scale
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            skeleton = skeleton * scale

        # Random temporal shift (roll along time axis)
        if np.random.rand() < 0.3:
            shift = np.random.randint(-5, 6)
            skeleton = np.roll(skeleton, shift, axis=0)

        return skeleton
