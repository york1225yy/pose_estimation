"""Skeleton graph definition for OpenPose 25-body keypoints."""

import numpy as np

# 25 OpenPose body keypoints (excluding 'background')
JOINT_NAMES = [
    'nose', 'neck', 'rShoulder', 'rElbow', 'rWrist',
    'lShoulder', 'lElbow', 'lWrist', 'rHip', 'rKnee',
    'rAnkle', 'lHip', 'lKnee', 'lAnkle', 'rEye',
    'lEye', 'rEar', 'lEar', 'lBigToe', 'lSmallToe',
    'lHeel', 'rBigToe', 'rSmallToe', 'rHeel', 'midHip'
]

NUM_JOINTS = len(JOINT_NAMES)  # 25

# OpenPose body-25 skeleton edges (parent, child)
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # nose -> neck -> rShoulder -> rElbow -> rWrist
    (1, 5), (5, 6), (6, 7),                  # neck -> lShoulder -> lElbow -> lWrist
    (1, 24), (24, 8), (8, 9), (9, 10),       # neck -> midHip -> rHip -> rKnee -> rAnkle
    (24, 11), (11, 12), (12, 13),             # midHip -> lHip -> lKnee -> lAnkle
    (0, 14), (0, 15), (14, 16), (15, 17),    # nose -> eyes -> ears
    (10, 23), (10, 21), (21, 22),             # rAnkle -> rHeel, rBigToe -> rSmallToe
    (13, 20), (13, 18), (18, 19),             # lAnkle -> lHeel, lBigToe -> lSmallToe
]

# Center joint for spatial graph partitioning
CENTER_JOINT = 1  # neck


def get_adjacency_matrix(num_joints=NUM_JOINTS, edges=EDGES):
    """Build symmetric adjacency matrix with self-loops."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # self-loops
    for i in range(num_joints):
        A[i, i] = 1.0
    return A


def normalize_adjacency(A):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.zeros_like(D)
    mask = D > 0
    D_inv_sqrt[mask] = 1.0 / np.sqrt(D[mask])
    D_inv_sqrt = np.diag(D_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def get_spatial_graph(strategy='distance'):
    """
    Return normalized adjacency matrices for ST-GCN.

    strategy:
        'uniform': single identity-like adjacency
        'distance': partition by hop distance (self, neighbors)
        'spatial': partition by distance to center (closer, farther, self)
    """
    A = get_adjacency_matrix()

    if strategy == 'uniform':
        return np.expand_dims(normalize_adjacency(A), axis=0)

    elif strategy == 'distance':
        # Partition: self-loop vs. neighbor
        I = np.eye(NUM_JOINTS, dtype=np.float32)
        A_neighbor = A - I
        A_list = [normalize_adjacency(I), normalize_adjacency(A_neighbor)]
        return np.stack(A_list, axis=0)

    elif strategy == 'spatial':
        I = np.eye(NUM_JOINTS, dtype=np.float32)
        A_raw = A - I  # pure adjacency without self-loops

        # Compute hop distance from center
        from scipy.sparse.csgraph import shortest_path
        dist = shortest_path(A_raw, directed=False)

        A_closer = np.zeros_like(A_raw)
        A_further = np.zeros_like(A_raw)

        for i in range(NUM_JOINTS):
            for j in range(NUM_JOINTS):
                if A_raw[i, j] == 0:
                    continue
                if dist[j, CENTER_JOINT] <= dist[i, CENTER_JOINT]:
                    A_closer[i, j] = 1.0
                else:
                    A_further[i, j] = 1.0

        A_list = [
            normalize_adjacency(I),
            normalize_adjacency(A_closer),
            normalize_adjacency(A_further),
        ]
        return np.stack(A_list, axis=0)

    raise ValueError(f"Unknown strategy: {strategy}")
