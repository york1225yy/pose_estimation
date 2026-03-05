"""
骨骼关键点可视化 & 坐姿标注模块
支持在图像 / 视频帧上绘制：
  - 17 个 COCO 关键点 + 骨骼连线
  - 坐姿结果文字标注
  - 几何参数面板（可选）
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

from .posture_rules import (
    SKELETON, KEYPOINT_NAMES, PostureParams,
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST, L_HIP, R_HIP,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)


# ── 色彩方案 ──────────────────────────────────────────────────
# BGR 格式
KP_COLORS = {
    # 面部：黄色
    NOSE: (0, 255, 255), L_EYE: (0, 255, 255), R_EYE: (0, 255, 255),
    L_EAR: (0, 200, 255), R_EAR: (0, 200, 255),
    # 上肢：绿色
    L_SHOULDER: (0, 255, 0), R_SHOULDER: (0, 255, 0),
    L_ELBOW: (0, 200, 0), R_ELBOW: (0, 200, 0),
    L_WRIST: (0, 180, 0), R_WRIST: (0, 180, 0),
    # 躯干/下肢：蓝色
    L_HIP: (255, 128, 0), R_HIP: (255, 128, 0),
    L_KNEE: (255, 0, 0), R_KNEE: (255, 0, 0),
    L_ANKLE: (255, 0, 128), R_ANKLE: (255, 0, 128),
}

LIMB_COLORS = {
    # 面部
    (NOSE, L_EYE): (0, 255, 255), (NOSE, R_EYE): (0, 255, 255),
    (L_EYE, L_EAR): (0, 230, 255), (R_EYE, R_EAR): (0, 230, 255),
    # 耳→肩
    (L_EAR, L_SHOULDER): (0, 200, 200), (R_EAR, R_SHOULDER): (0, 200, 200),
    # 肩横
    (L_SHOULDER, R_SHOULDER): (0, 255, 0),
    # 上肢
    (L_SHOULDER, L_ELBOW): (80, 255, 80), (R_SHOULDER, R_ELBOW): (80, 255, 80),
    (L_ELBOW, L_WRIST): (50, 200, 50), (R_ELBOW, R_WRIST): (50, 200, 50),
    # 躯干
    (L_SHOULDER, L_HIP): (255, 200, 0), (R_SHOULDER, R_HIP): (255, 200, 0),
    # 髋横
    (L_HIP, R_HIP): (255, 128, 0),
    # 下肢
    (L_HIP, L_KNEE): (255, 80, 0), (R_HIP, R_KNEE): (255, 80, 0),
    (L_KNEE, L_ANKLE): (255, 0, 80), (R_KNEE, R_ANKLE): (255, 0, 80),
}

# Posture → bounding box / text color
POSTURE_COLOR_MAP = {
    "Normal Driving":  (0, 255, 0),     # green
    "Slight Lean Fwd": (0, 220, 255),   # yellow
    "Heavy Lean Fwd":  (0, 128, 255),   # orange
    "Recline Relax":   (255, 255, 0),   # cyan
    "Side Reach":      (255, 128, 0),   # light blue
    "Semi-Reclined":   (0, 0, 255),     # red
    "Head Down Phone": (0, 80, 255),    # orange-red
    "Sleeping":        (0, 0, 200),     # dark red
    "Lateral Recline": (128, 0, 255),   # purple
    "Upright Working": (255, 200, 100), # teal
    "Unknown":         (180, 180, 180),
}


class PoseVisualizer:
    """在 OpenCV 帧上绘制骨骼关键点 & 坐姿标注"""

    def __init__(self,
                 conf_threshold: float = 0.4,
                 kp_radius: int = 5,
                 limb_thickness: int = 2,
                 show_params: bool = True,
                 font_scale: float = 0.55):
        """
        Parameters
        ----------
        conf_threshold : float
            低置信度关键点不绘制
        kp_radius : int
            关键点圆半径
        limb_thickness : int
            骨骼线粗细
        show_params : bool
            是否在画面右侧显示参数面板
        font_scale : float
            字体缩放
        """
        self.conf_thr = conf_threshold
        self.kp_radius = kp_radius
        self.limb_thick = limb_thickness
        self.show_params = show_params
        self.font_scale = font_scale

    def draw_skeleton(self, frame: np.ndarray,
                      keypoints: np.ndarray,
                      confidences: np.ndarray) -> np.ndarray:
        """在帧上绘制单人骨骼"""
        kp = keypoints.astype(int)
        conf = confidences

        # 按图像高度自适应：以 480px 为基准
        scale = frame.shape[0] / 480.0
        radius = max(3, int(self.kp_radius * scale))
        thick  = max(1, int(self.limb_thick * scale))

        # 绘制骨骼线
        for (i, j) in SKELETON:
            if conf[i] > self.conf_thr and conf[j] > self.conf_thr:
                color = LIMB_COLORS.get((i, j), (200, 200, 200))
                cv2.line(frame, tuple(kp[i]), tuple(kp[j]),
                         color, thick, cv2.LINE_AA)

        # 绘制关键点
        for i in range(17):
            if conf[i] > self.conf_thr:
                color = KP_COLORS.get(i, (255, 255, 255))
                cv2.circle(frame, tuple(kp[i]), radius,
                           color, -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(kp[i]), radius,
                           (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def draw_posture_label(self, frame: np.ndarray,
                           bbox: Optional[Tuple[int, int, int, int]],
                           params: PostureParams,
                           person_idx: int = 0) -> np.ndarray:
        """
        在检测框上方绘制坐姿标签。

        Parameters
        ----------
        bbox : tuple (x1, y1, x2, y2) or None
        params : PostureParams
        person_idx : int
            多人时的编号
        """
        label = params.posture_label
        color = POSTURE_COLOR_MAP.get(label, (180, 180, 180))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 按图像高度自适应字体
        scale = frame.shape[0] / 480.0
        fs = self.font_scale * scale
        thick = max(1, int(2 * scale))

        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, int(2 * scale)))
            # 文字背景
            txt = f"P{person_idx}: {label}"
            (tw, th), _ = cv2.getTextSize(txt, font, fs, thick)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, txt, (x1 + 3, y1 - 5),
                        font, fs, (255, 255, 255), thick, cv2.LINE_AA)
        else:
            txt = f"P{person_idx}: {label}"
            cv2.putText(frame, txt, (10, 30 + person_idx * int(35 * scale)),
                        font, fs, color, thick, cv2.LINE_AA)

        return frame

    def draw_param_panel(self, frame: np.ndarray,
                         params: PostureParams,
                         person_idx: int = 0) -> np.ndarray:
        """在画面左上角绘制半透明参数面板"""
        if not self.show_params:
            return frame

        # 只显示当前 4 类坐姿的关键判别参数（数据驱动阈值）：
        #   Recline Relax   → Trunk Tilt > 24°
        #   Heavy Lean Fwd  → Trunk Tilt < 13° & Head Fwd ≤ 35°
        #   Head Down Phone → Trunk Tilt < 13° & Head Fwd > 35°
        #   Normal Driving  → Trunk Tilt 13~24°
        lines = [
            f"[P{person_idx}] Posture: {params.posture_label}",
            f"Trunk Tilt: {params.trunk_tilt:.1f} deg  (<13 lean | 13-24 normal | >24 recline)",
            f"Head Fwd:   {params.head_forward_angle:.1f} deg  (>35 -> Phone)",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX

        # 按图像高度自适应（基准 480px）
        h_scale = frame.shape[0] / 480.0
        fs_candidate = self.font_scale * 0.85 * h_scale

        # 额外的宽度约束：保证最长行能放进画面（留 20px 边距）
        avail_w = frame.shape[1] - 20
        fs = fs_candidate
        for ln in lines:
            (tw, _), _ = cv2.getTextSize(ln, font, fs_candidate, 1)
            if tw > avail_w and tw > 0:
                fs = min(fs, fs_candidate * avail_w / tw)
        fs = max(0.3, fs)  # 字体不低于 0.3，保证基本可读

        line_h = max(18, int(26 * h_scale * (fs / fs_candidate if fs_candidate > 0 else 1)))
        pad    = max(6,  int(10 * h_scale))
        y_start = pad + person_idx * (len(lines) * line_h + 2 * pad + pad)

        # 计算面板宽度
        max_w = 0
        for ln in lines:
            (tw, _), _ = cv2.getTextSize(ln, font, fs, 1)
            max_w = max(max_w, tw)

        # 半透明背景
        overlay = frame.copy()
        x1, y1 = 8, y_start
        x2 = x1 + max_w + 2 * pad
        y2 = y1 + len(lines) * line_h + 2 * pad
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 写文字
        for i, ln in enumerate(lines):
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            cv2.putText(frame, ln, (x1 + pad, y1 + pad + (i + 1) * line_h - 5),
                        font, fs, color, 1, cv2.LINE_AA)

        return frame

    def draw(self, frame: np.ndarray,
             keypoints: np.ndarray,
             confidences: np.ndarray,
             params: PostureParams,
             bbox: Optional[Tuple[int, int, int, int]] = None,
             person_idx: int = 0) -> np.ndarray:
        """
        一步完成：骨骼 + 标签 + 参数面板
        """
        frame = self.draw_skeleton(frame, keypoints, confidences)
        frame = self.draw_posture_label(frame, bbox, params, person_idx)
        frame = self.draw_param_panel(frame, params, person_idx)
        return frame
