"""
SegmentDecordInit：结合 decord 视频初始化与片段裁剪。

在标准 DecordInit 之后，覆盖 total_frames 和 start_index，
使后续 SampleFrames 只在 [segment_start, segment_end) 范围内采样。
"""
import io

import numpy as np
from mmcv.fileio import FileClient

from ..builder import PIPELINES


@PIPELINES.register_module()
class SegmentDecordInit:
    """用 decord 初始化视频，并将采样范围限制到指定帧段。

    results 中需包含:
        - ``filename``: 视频文件路径
        - ``segment_start`` (int): 片段起始帧（含）
        - ``segment_end``   (int): 片段结束帧（不含）

    初始化后会在 results 中设置:
        - ``video_reader``: decord.VideoReader 对象
        - ``total_frames``: segment_end - segment_start（片段长度）
        - ``start_index``:  segment_start（绝对帧偏移）

    Args:
        io_backend (str): 文件读取后端，默认 'disk'。
        num_threads (int): decord 解码线程数，默认 1。
    """

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        try:
            import decord
        except ImportError:
            raise ImportError('请先执行 "pip install decord" 安装 decord。')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)

        total_video_frames = len(container)

        seg_start = int(results.get('segment_start', 0))
        seg_end = int(results.get('segment_end', total_video_frames))

        # 防止越界
        seg_start = max(0, min(seg_start, total_video_frames - 1))
        seg_end = max(seg_start + 1, min(seg_end, total_video_frames))

        results['video_reader'] = container
        # 覆盖 total_frames 与 start_index，让 SampleFrames 在片段范围内采样
        results['total_frames'] = seg_end - seg_start
        results['start_index'] = seg_start

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'io_backend={self.io_backend}, '
                f'num_threads={self.num_threads})')
