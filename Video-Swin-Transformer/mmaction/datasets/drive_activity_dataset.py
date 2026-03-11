import copy
import os.path as osp

import pandas as pd

from .base import BaseDataset
from .builder import DATASETS

# 目标5类活动，编号 0-4
TARGET_CLASSES = [
    'sitting_still',
    'eating',
    'fetching_an_object',
    'placing_an_object',
    'reading_magazine',
]

LABEL_MAP = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}


@DATASETS.register_module()
class DriveActivityDataset(BaseDataset):
    """驾驶舱活动识别数据集，从 midlevel.chunks_90.csv 中读取分段视频标注。

    CSV 格式::

        participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id

    其中 ``file_id`` 形如 ``vp1/run1b_2018-05-29-14-02-47.ids_1``，
    对应视频文件 ``<data_prefix>/<file_id>.mp4``。

    Args:
        ann_file (str): CSV 标注文件路径。
        pipeline (list[dict]): 数据变换流水线。
        data_prefix (str): 视频文件根目录（应包含 vp1/, vp2/ 等子目录）。
        start_index (int): 帧索引偏移，视频输入时固定为 0。
        **kwargs: 传给 BaseDataset 的其余参数。
    """

    def __init__(self, ann_file, pipeline, data_prefix=None,
                 start_index=0, **kwargs):
        super().__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            start_index=start_index,
            **kwargs,
        )

    def load_annotations(self):
        """从 CSV 文件加载视频片段信息，只保留目标5类。"""
        df = pd.read_csv(self.ann_file)

        # 只保留目标类别
        df = df[df['activity'].isin(TARGET_CLASSES)].reset_index(drop=True)

        video_infos = []
        for _, row in df.iterrows():
            # file_id 示例: "vp1/run1b_2018-05-29-14-02-47.ids_1"
            file_id = row['file_id']
            video_filename = f"{file_id}.mp4"

            if self.data_prefix is not None:
                filename = osp.join(self.data_prefix, video_filename)
            else:
                filename = video_filename

            video_infos.append(dict(
                filename=filename,
                label=LABEL_MAP[row['activity']],
                segment_start=int(row['frame_start']),
                segment_end=int(row['frame_end']),
            ))

        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def evaluate(self, results, metrics='top_k_accuracy', metric_options=None,
                 logger=None):
        if metric_options is None:
            metric_options = {'top_k_accuracy': {'topk': (1, 5)}}
        return super().evaluate(results, metrics=metrics,
                                metric_options=metric_options, logger=logger)
