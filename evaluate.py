"""
Standalone evaluation script for ST-GCN skeleton action recognition.

Loads a saved model checkpoint and computes full metrics on the test set
(same 70/15/15 random split as training, seed=42).

Usage:
    # Evaluate best_model.pth on test set (default)
    python evaluate.py

    # Evaluate a specific checkpoint on val set
    python evaluate.py --model output/last_model.pth --split val

    # Evaluate on all three splits (train / val / test)
    python evaluate.py --split all

    # Custom data / model paths
    python evaluate.py --model output/best_model.pth --pose_dir pose_all --batch_size 64
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from stgcn.model import STGCN
from stgcn.dataset import PoseDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(BASE_DIR, 'activity_label')
POSE_DIR  = os.path.join(BASE_DIR, 'pose_all')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='ST-GCN Evaluation')
    parser.add_argument('--model', type=str, default='output/best_model.pth',
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which data split to evaluate (default: test)')
    parser.add_argument('--pose_dir', type=str, default=POSE_DIR,
                        help='Directory containing *.openpose.3d.csv files')
    parser.add_argument('--label_csv', type=str,
                        default=os.path.join(LABEL_DIR, 'tasklevel.chunks_90.csv'),
                        help='Full label CSV path')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                        help='Subset of class names to evaluate (must match training classes)')
    parser.add_argument('--max_frames', type=int, default=90)
    parser.add_argument('--graph_strategy', type=str, default='spatial',
                        choices=['uniform', 'distance', 'spatial'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42,
                        help='Must match the seed used during training')
    parser.add_argument('--save', action='store_true',
                        help='Save results to output/eval_results.json')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_splits(args):
    """Reproduce the exact same 70/15/15 random split used in train.py."""
    full_dataset = PoseDataset(
        args.label_csv, args.pose_dir,
        max_frames=args.max_frames, augment=False,
        allowed_classes=args.classes,
    )
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Dataset: {n_total} samples  →  train={n_train}, val={n_val}, test={n_test}")
    return {'train': train_set, 'val': val_set, 'test': test_set}, full_dataset.activity_labels, full_dataset.num_classes


def make_loader(dataset, args, device):
    pin = (device.type == 'cuda')
    nw  = args.num_workers if device.type == 'cuda' else 0
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=nw, pin_memory=pin)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    """Returns (all_labels, all_preds, all_probs)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for skeletons, labels in loader:
        skeletons = skeletons.to(device, non_blocking=True)
        logits = model(skeletons)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),   # shape (N, num_classes)
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, preds, probs):
    """Compute the full set of classification metrics."""
    metrics = {
        'accuracy':          accuracy_score(labels, preds),
        'precision_macro':   precision_score(labels, preds, average='macro',    zero_division=0),
        'precision_weighted':precision_score(labels, preds, average='weighted', zero_division=0),
        'recall_macro':      recall_score(labels, preds, average='macro',    zero_division=0),
        'recall_weighted':   recall_score(labels, preds, average='weighted', zero_division=0),
        'f1_macro':          f1_score(labels, preds, average='macro',    zero_division=0),
        'f1_weighted':       f1_score(labels, preds, average='weighted', zero_division=0),
        'f1_per_class':      f1_score(labels, preds, average=None,       zero_division=0).tolist(),
    }

    # Per-class softmax confidence for correctly / incorrectly predicted samples
    correct_mask = (labels == preds)
    if correct_mask.any():
        metrics['mean_confidence_correct'] = float(
            probs[correct_mask, preds[correct_mask]].mean()
        )
    if (~correct_mask).any():
        metrics['mean_confidence_wrong'] = float(
            probs[~correct_mask, preds[~correct_mask]].mean()
        )

    return metrics


def print_results(split_name, metrics, labels, preds, activity_labels):
    num_classes = len(activity_labels)
    sep = '=' * 70
    print(f"\n{sep}")
    print(f"  Split: {split_name.upper()}   |   Samples: {len(labels)}")
    print(sep)

    print(f"  Accuracy          : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (macro) : {metrics['precision_macro']:.4f}")
    print(f"  Precision (wt.)   : {metrics['precision_weighted']:.4f}")
    print(f"  Recall    (macro) : {metrics['recall_macro']:.4f}")
    print(f"  Recall    (wt.)   : {metrics['recall_weighted']:.4f}")
    print(f"  F1        (macro) : {metrics['f1_macro']:.4f}")
    print(f"  F1        (wt.)   : {metrics['f1_weighted']:.4f}")
    if 'mean_confidence_correct' in metrics:
        print(f"  Avg confidence (correct predictions) : {metrics['mean_confidence_correct']:.4f}")
    if 'mean_confidence_wrong' in metrics:
        print(f"  Avg confidence (wrong   predictions) : {metrics['mean_confidence_wrong']:.4f}")

    # Per-class F1
    print(f"\n  {'Class':<26}  F1")
    print(f"  {'-'*35}")
    for i, f1 in enumerate(metrics['f1_per_class']):
        marker = '  <<' if f1 == min(metrics['f1_per_class']) else ''
        print(f"  {activity_labels[i]:<26}  {f1:.4f}{marker}")

    # Classification report
    present = sorted(set(labels))
    target_names = [activity_labels[i] for i in present]
    report = classification_report(
        labels, preds,
        labels=present,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )
    print(f"\n  Classification Report:\n")
    for line in report.splitlines():
        print(f"  {line}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    short = [n[:8] for n in activity_labels]
    print(f"\n  Confusion Matrix (rows=true, cols=pred):\n")
    header = "           " + "".join(f"{n:>10s}" for n in short)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:10d}" for v in row)
        print(f"  {short[i]:>9s}  {row_str}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # Model
    model_path = args.model if os.path.isabs(args.model) \
        else os.path.join(BASE_DIR, args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = STGCN(
        num_classes=num_classes,
        in_channels=3,
        graph_strategy=args.graph_strategy,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    print(f"Loaded : {model_path}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    # Data splits
    splits, activity_labels, num_classes = build_splits(args)

    # Which splits to evaluate
    eval_splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    all_results = {}
    for split_name in eval_splits:
        loader = make_loader(splits[split_name], args, device)
        labels, preds, probs = run_inference(model, loader, device)
        metrics = compute_metrics(labels, preds, probs)
        print_results(split_name, metrics, labels, preds, activity_labels)
        all_results[split_name] = metrics

    # Optionally save
    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, 'eval_results.json')
        with open(out_path, 'w') as f:
            json.dump({'model': model_path, 'results': all_results}, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
