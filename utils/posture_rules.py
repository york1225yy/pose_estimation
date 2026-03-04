"""
驾驶员坐姿分类规则引擎
基于 COCO 17 关键点体系，通过几何角度/距离特征判断 10 类驾驶坐姿。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


# ── COCO 17 关键点索引 ──────────────────────────────────────────
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# 骨骼连接关系（用于可视化）
SKELETON = [
    (NOSE, L_EYE), (NOSE, R_EYE),
    (L_EYE, L_EAR), (R_EYE, R_EAR),
    (L_EAR, L_SHOULDER), (R_EAR, R_SHOULDER),
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW),
    (L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (R_HIP, R_KNEE),
    (L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE),
]

KEYPOINT_NAMES = [
    "鼻", "左眼", "右眼", "左耳", "右耳",
    "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
    "左髋", "右髋", "左膝", "右膝", "左踝", "右踝",
]

# 10 种坐姿标签
POSTURE_LABELS = [
    "标准驾驶坐姿", "轻微前倾", "严重前倾", "后仰放松",
    "侧身取物", "半躺休息", "低头玩手机", "睡姿",
    "斜躺副驾", "直立办公",
]

# 参考向量（图像坐标系 Y 轴向下）
VERTICAL_UP = np.array([0.0, -1.0])
HORIZONTAL = np.array([1.0, 0.0])


def _angle_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个二维向量的夹角（度），范围 [0, 180]"""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


@dataclass
class PostureParams:
    """每帧坐姿几何参数"""
    trunk_tilt: float = 0.0          # 躯干前倾角（度）
    trunk_backward: bool = False     # 是否后仰
    head_forward_angle: float = 0.0  # 头前倾角（度）
    head_side_angle: float = 0.0     # 头侧倾角（度）
    trunk_rotate_angle: float = 0.0  # 躯干旋转角（度）
    shoulder_lift_norm: float = 0.0  # 单侧肩膀抬起量（归一化）
    arm_raised: bool = False         # 手臂是否抬起
    l_arm_up: bool = False           # 左臂抬起
    r_arm_up: bool = False           # 右臂抬起
    head_backward_angle: float = 0.0 # 头后仰角（度）
    lateral_lean: float = 0.0        # 侧向倾斜量
    posture_label: str = "未知"      # 最终判定的坐姿
    confidence_valid: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "躯干前倾角(度)": round(self.trunk_tilt, 2),
            "躯干后仰": self.trunk_backward,
            "头前倾角(度)": round(self.head_forward_angle, 2),
            "头侧倾角(度)": round(self.head_side_angle, 2),
            "躯干旋转角(度)": round(self.trunk_rotate_angle, 2),
            "肩膀抬起量(归一化)": round(self.shoulder_lift_norm, 3),
            "手臂抬起": self.arm_raised,
            "头后仰角(度)": round(self.head_backward_angle, 2),
            "侧向倾斜量": round(self.lateral_lean, 3),
            "判定坐姿": self.posture_label,
        }
        return d

    def to_print_str(self) -> str:
        lines = [
            f"  躯干前倾角: {self.trunk_tilt:.1f}°  ({'后仰' if self.trunk_backward else '前倾'})",
            f"  头前倾角:   {self.head_forward_angle:.1f}°",
            f"  头侧倾角:   {self.head_side_angle:.1f}°",
            f"  头后仰角:   {self.head_backward_angle:.1f}°",
            f"  躯干旋转角: {self.trunk_rotate_angle:.1f}°",
            f"  肩膀抬起量: {self.shoulder_lift_norm:.3f}",
            f"  手臂抬起:   {'是' if self.arm_raised else '否'} (左:{self.l_arm_up} 右:{self.r_arm_up})",
            f"  侧向倾斜:   {self.lateral_lean:.3f}",
            f"  ──────────────────────────",
            f"  判定坐姿:   【{self.posture_label}】",
        ]
        return "\n".join(lines)


class PostureClassifier:
    """驾驶员坐姿分类器（规则引擎）"""

    def __init__(self, conf_threshold: float = 0.4, seat_back_angle: Optional[float] = None):
        """
        Parameters
        ----------
        conf_threshold : float
            关键点置信度阈值，低于此值的关键点被视为不可用
        seat_back_angle : float or None
            靠背角度传感器读数（度），若无传感器则为 None
        """
        self.conf_thr = conf_threshold
        self.seat_back_angle = seat_back_angle

    def _valid(self, conf: np.ndarray, *idxs) -> bool:
        """检查给定关键点是否全部有效"""
        return all(conf[i] > self.conf_thr for i in idxs)

    def classify(self, keypoints: np.ndarray, confidences: np.ndarray,
                 seat_back_angle: Optional[float] = None) -> PostureParams:
        """
        对单人关键点进行坐姿分类。

        Parameters
        ----------
        keypoints : np.ndarray, shape (17, 2)
            COCO 17 关键点的像素坐标 (x, y)
        confidences : np.ndarray, shape (17,)
            各关键点置信度 [0, 1]
        seat_back_angle : float or None
            当前帧的靠背角度（覆盖默认值）

        Returns
        -------
        PostureParams
            包含所有几何参数和最终坐姿标签
        """
        kp = keypoints
        conf = confidences
        sba = seat_back_angle if seat_back_angle is not None else self.seat_back_angle
        params = PostureParams()

        # ── 记录有效性 ──
        params.confidence_valid = {
            "nose": self._valid(conf, NOSE),
            "ears": self._valid(conf, L_EAR, R_EAR),
            "shoulders": self._valid(conf, L_SHOULDER, R_SHOULDER),
            "elbows": self._valid(conf, L_ELBOW, R_ELBOW),
            "hips": self._valid(conf, L_HIP, R_HIP),
        }

        # ── 基础几何点 ──
        neck = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2.0
        mid_hip = (kp[L_HIP] + kp[R_HIP]) / 2.0
        mid_ear = (kp[L_EAR] + kp[R_EAR]) / 2.0

        # ── 向量 ──
        trunk_vec = neck - mid_hip       # 躯干轴：髋 → 肩
        neck_vec = mid_ear - neck        # 颈轴：  肩 → 耳
        nose_vec = kp[NOSE] - neck       # 鼻向量：肩 → 鼻

        # ══ 1. 躯干前倾角 ══
        params.trunk_tilt = _angle_2d(trunk_vec, VERTICAL_UP)
        params.trunk_backward = trunk_vec[1] > 0  # Y 分量正 = 后仰

        # ══ 2. 头前倾角 ══
        params.head_forward_angle = _angle_2d(neck_vec, trunk_vec)

        # ══ 3. 头侧倾角 ══
        if self._valid(conf, L_EAR, R_EAR):
            ear_vec = kp[L_EAR] - kp[R_EAR]
            ear_angle = _angle_2d(ear_vec, HORIZONTAL)
            params.head_side_angle = abs(90.0 - ear_angle)
        else:
            params.head_side_angle = 0.0

        # ══ 4. 躯干旋转角 ══
        shoulder_dist_2d = abs(kp[L_SHOULDER][0] - kp[R_SHOULDER][0])
        shoulder_dist_3d = np.linalg.norm(kp[L_SHOULDER] - kp[R_SHOULDER])
        if shoulder_dist_3d > 1e-6:
            ratio = np.clip(shoulder_dist_2d / shoulder_dist_3d, 0.0, 1.0)
            params.trunk_rotate_angle = float(np.degrees(np.arccos(ratio)))
        else:
            params.trunk_rotate_angle = 0.0

        # ══ 5. 单侧肩膀抬起 ══
        if shoulder_dist_3d > 1e-6:
            params.shoulder_lift_norm = abs(kp[L_SHOULDER][1] - kp[R_SHOULDER][1]) / shoulder_dist_3d
        else:
            params.shoulder_lift_norm = 0.0

        # ══ 6. 手臂抬起 ══
        params.l_arm_up = (self._valid(conf, L_ELBOW, L_SHOULDER)
                           and kp[L_ELBOW][1] < kp[L_SHOULDER][1])
        params.r_arm_up = (self._valid(conf, R_ELBOW, R_SHOULDER)
                           and kp[R_ELBOW][1] < kp[R_SHOULDER][1])
        params.arm_raised = params.l_arm_up or params.r_arm_up

        # ══ 7. 头后仰角 ══
        if np.linalg.norm(trunk_vec) > 1e-6:
            params.head_backward_angle = _angle_2d(neck_vec, -trunk_vec)
        else:
            params.head_backward_angle = 0.0

        # ══ 8. 侧向倾斜 ══
        if shoulder_dist_3d > 1e-6:
            params.lateral_lean = (kp[R_SHOULDER][1] - kp[L_SHOULDER][1]) / shoulder_dist_3d
        else:
            params.lateral_lean = 0.0

        # ══════════════════════════════════════════════════════
        # 规则引擎（优先级从高到低）
        # ══════════════════════════════════════════════════════

        # 规则 1：睡姿 — 头大幅度侧倾 > 35°
        if self._valid(conf, L_EAR, R_EAR) and params.head_side_angle > 35:
            params.posture_label = "睡姿"
            return params

        # 规则 2：半躺休息 / 斜躺副驾 — 靠背 ≥ 120° 或躯干后仰 > 35°
        reclined = ((sba is not None and sba >= 120) or
                    (params.trunk_backward and params.trunk_tilt > 35))
        if reclined and self._valid(conf, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            if params.lateral_lean > 0.15:
                params.posture_label = "斜躺副驾"
            else:
                params.posture_label = "半躺休息"
            return params

        # 规则 3：侧身取物 — 躯干旋转 > 15° 且单侧肩膀抬高 > 20%
        if (self._valid(conf, L_SHOULDER, R_SHOULDER)
                and params.trunk_rotate_angle > 15
                and params.shoulder_lift_norm > 0.20):
            params.posture_label = "侧身取物"
            return params

        # 规则 4：低头玩手机 — 头前倾 > 25°
        if self._valid(conf, NOSE, L_SHOULDER, R_SHOULDER) and params.head_forward_angle > 25:
            params.posture_label = "低头玩手机"
            return params

        # 规则 5：严重前倾 — 躯干前倾 > 20°
        if (self._valid(conf, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
                and not params.trunk_backward and params.trunk_tilt > 20):
            params.posture_label = "严重前倾"
            return params

        # 规则 6：轻微前倾 — 躯干前倾 8°~20°
        if (self._valid(conf, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
                and not params.trunk_backward and 8 < params.trunk_tilt <= 20):
            params.posture_label = "轻微前倾"
            return params

        # 规则 7：直立办公 — 背挺直（< 5°）且手臂抬起
        if params.trunk_tilt < 5 and params.arm_raised:
            params.posture_label = "直立办公"
            return params

        # 规则 8：后仰放松 — 头后仰 10°~25°，躯干直立 < 8°
        if self._valid(conf, NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER):
            if 10 <= params.head_backward_angle <= 25 and params.trunk_tilt < 8:
                params.posture_label = "后仰放松"
                return params

        # 规则 9（默认）：标准驾驶坐姿
        params.posture_label = "标准驾驶坐姿"
        return params
