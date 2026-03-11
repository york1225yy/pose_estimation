"""
Video-based Action Recognition — Evaluation Script

Loads a saved TransDARC checkpoint and computes full metrics on any split.
Reproduces the exact same 70/15/15 random split used in video_train.py.

Usage:
    # Evaluate best_transdarc.pth on test set
    python video_evaluate.py

    # Evaluate on val set
    python video_evaluate.py --split val

    # Custom checkpoint / data
    python video_evaluate.py \\
        --model output/best_transdarc.pth \\
        --label_csv activity_label/tasklevel.chunks_90.csv \\
        --classes sitting_still eating fetching_an_object placing_an_object reading_magazine

    # Evaluate on all splits
    python video_evaluate.py --split all --save
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from video.dataset import VideoDataset
from video.transdamc import TransDARC

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR  = os.path.join(BASE_DIR, 'video_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

MODEL_REGISTRY = {
    'transdarc': TransDARC,
}

DEFAULT_LABEL_CSV = os.path.join(BASE_DIR, 'activity_label', 'tasklevel.chunks_90.csv')
DEFAULT_5_CLASSES = [
    'sitting_still', 'eating', 'fetching_an_object',
    'placing_an_object', 'reading_magazine',
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Video Action Recognition Evaluation (TransDARC)')
    parser.add_argument('--arch', type=str, default='transdarc',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture (default: transdarc)')
    parser.add_argument('--model', type=str, default=None,
                        help='Checkpoint path. Default: output/best_<arch>.pth')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to evaluate (default: test)')
    parser.add_argument('--label_csv', type=str, default=DEFAULT_LABEL_CSV,
                        help='Path to activity label CSV file')
    parser.add_argument('--classes', type=str, nargs='+', default=DEFAULT_5_CLASSES,
                        help='Class names (must match training)')
    parser.add_argument('--video_dir', type=str, default=VIDEO_DIR,
                        help='Directory containing *.mp4 files')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42,
                        help='Must match seed used in video_train.py')
    parser.add_argument('--save', action='store_true',
                        help='Save results to output/eval_video_<arch>.json')
    return parser.parse_args()


# Video Swin official participant split
_TRAIN_VPS   = {1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15}
_VALTEST_VPS = {8, 9, 14}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_splits(args):
    full_dataset = VideoDataset(
        label_csv=args.label_csv,
        video_dir=args.video_dir,
        allowed_classes=args.classes,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        augment=False,
    )
    n_total = len(full_dataset)

    pids = full_dataset.samples['participant_id'].astype(int).values
    train_idx   = [i for i, p in enumerate(pids) if p in _TRAIN_VPS]
    valtest_idx = [i for i, p in enumerate(pids) if p in _VALTEST_VPS]

    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, valtest_idx)
    test_set  = Subset(full_dataset, valtest_idx)

    print(f"Dataset: {n_total} clips → "
          f"train={len(train_idx)} (vp {sorted(_TRAIN_VPS)}), "
          f"val=test={len(valtest_idx)} "
          f"(vp {sorted(_VALTEST_VPS)})")
    return (
        {'train': train_set, 'val': val_set, 'test': test_set},
        full_dataset.activity_labels,
        full_dataset.num_classes,
    )


def make_loader(dataset, args, device):
    pin = device.type == 'cuda'
    nw  = args.num_workers if device.type == 'cuda' else 0
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=nw, pin_memory=pin)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True)
        logits = model(clips)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ---------------------------------------------------------------------------
# Metrics & display
# ---------------------------------------------------------------------------

def compute_metrics(labels, preds, probs):
    metrics = {
        'accuracy':           accuracy_score(labels, preds),
        'precision_macro':    precision_score(labels, preds, average='macro',    zero_division=0),
        'precision_weighted': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall_macro':       recall_score   (labels, preds, average='macro',    zero_division=0),
        'recall_weighted':    recall_score   (labels, preds, average='weighted', zero_division=0),
        'f1_macro':           f1_score       (labels, preds, average='macro',    zero_division=0),
        'f1_weighted':        f1_score       (labels, preds, average='weighted', zero_division=0),
        'f1_per_class':       f1_score       (labels, preds, average=None,       zero_division=0).tolist(),
    }
    correct_mask = (labels == preds)
    if correct_mask.any():
        metrics['mean_confidence_correct'] = float(
            probs[correct_mask, preds[correct_mask]].mean())
    if (~correct_mask).any():
        metrics['mean_confidence_wrong']   = float(
            probs[~correct_mask, preds[~correct_mask]].mean())
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
        print(f"  Avg conf (correct): {metrics['mean_confidence_correct']:.4f}")
    if 'mean_confidence_wrong' in metrics:
        print(f"  Avg conf (wrong)  : {metrics['mean_confidence_wrong']:.4f}")

    # Per-class F1
    print(f"\n  {'Class':<26}  F1")
    print(f"  {'-'*35}")
    min_f1 = min(metrics['f1_per_class'])
    for i, f1 in enumerate(metrics['f1_per_class']):
        marker = '  <<' if f1 == min_f1 else ''
        print(f"  {activity_labels[i]:<26}  {f1:.4f}{marker}")

    # Full classification report
    present      = sorted(set(labels))
    target_names = [activity_labels[i] for i in present]
    report = classification_report(labels, preds, labels=present,
                                   target_names=target_names, zero_division=0, digits=4)
    print(f"\n  Classification Report:\n")
    for line in report.splitlines():
        print(f"  {line}")

    # Confusion matrix
    cm    = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    short = [n[:8] for n in activity_labels]
    print(f"\n  Confusion Matrix (rows=true, cols=pred):\n")
    header = "           " + "".join(f"{n:>10s}" for n in short)
    print(f"  {header}")
    for i, row in enumerate(cm):
        print(f"  {short[i]:>9s}  " + "".join(f"{v:10d}" for v in row))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Resolve checkpoint
    if args.model is None:
        args.model = os.path.join('output', f'best_{args.arch}.pth')
    model_path = args.model if os.path.isabs(args.model) \
        else os.path.join(BASE_DIR, args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Data
    splits, activity_labels, num_classes = build_splits(args)

    # Model
    ModelClass = MODEL_REGISTRY[args.arch]
    model = ModelClass(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Arch   : {args.arch.upper()}")
    print(f"Loaded : {model_path}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    eval_splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    all_results = {}
    for split_name in eval_splits:
        loader = make_loader(splits[split_name], args, device)
        labels, preds, probs = run_inference(model, loader, device)
        metrics = compute_metrics(labels, preds, probs)
        print_results(split_name, metrics, labels, preds, activity_labels)
        all_results[split_name] = metrics

    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f'eval_video_{args.arch}.json')
        with open(out_path, 'w') as f:
            json.dump({'arch': args.arch, 'model': model_path,
                       'results': all_results}, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
