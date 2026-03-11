"""
Video-based Action Recognition — Training Script

Trains TransDARC (ICRA 2023) on video clips, using the same label CSVs
as the skeleton-based pipeline.

TransDARC trains with a Diffusion Probabilistic Model (DDPM) auxiliary loss
that augments latent-space features during training to improve generalisation.
At inference only the clean backbone + classifier is used.

Usage:
    # TransDARC with 5-class tasklevel subset
    python video_train.py \\
        --label_csv activity_label/tasklevel.chunks_90.csv \\
        --classes sitting_still eating fetching_an_object placing_an_object reading_magazine \\
        --epochs 30 --lr 1e-4 --batch_size 8

    # Without ImageNet pretrain; stronger DPM weight
    python video_train.py --no_pretrained --dpm_weight 2.0

Checkpoints are saved to:  output/best_transdarc.pth
                            output/last_transdarc.pth
Results JSON:               output/results_transdarc.json
History JSON:               output/history_transdarc.json
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
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
    parser = argparse.ArgumentParser(description='Video Action Recognition Training (TransDARC)')
    parser.add_argument('--arch', type=str, default='transdarc',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture (default: transdarc)')
    parser.add_argument('--label_csv', type=str, default=DEFAULT_LABEL_CSV,
                        help='Path to activity label CSV file')
    parser.add_argument('--classes', type=str, nargs='+', default=DEFAULT_5_CLASSES,
                        help='Class names to train on')
    parser.add_argument('--video_dir', type=str, default=VIDEO_DIR,
                        help='Directory containing *.mp4 files')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Frames sampled per clip (default: 8)')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Spatial crop size in pixels (default: 224)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4, AdamW)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dpm_weight', type=float, default=1.0,
                        help='Weight of DDPM auxiliary loss (default: 1.0; 0 = disable)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze pre-trained spatial backbone weights')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Train from scratch (no ImageNet init)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# Video Swin official participant split
_TRAIN_VPS   = {1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15}
_VALTEST_VPS = {8, 9, 14}


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def build_datasets(args):
    full_dataset = VideoDataset(
        label_csv=args.label_csv,
        video_dir=args.video_dir,
        allowed_classes=args.classes,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        augment=False,
    )
    n_total = len(full_dataset)
    if n_total == 0:
        raise RuntimeError("No samples found. Check --label_csv, --classes, and --video_dir.")

    pids = full_dataset.samples['participant_id'].astype(int).values
    train_idx   = [i for i, p in enumerate(pids) if p in _TRAIN_VPS]
    valtest_idx = [i for i, p in enumerate(pids) if p in _VALTEST_VPS]

    mid      = len(valtest_idx) // 2
    val_idx  = valtest_idx[:mid]
    test_idx = valtest_idx[mid:]

    train_sub = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    test_set  = Subset(full_dataset, test_idx)

    # Augmented wrapper for train split
    train_set = AugmentedVideoSubset(train_sub, full_dataset,
                                     num_frames=args.num_frames,
                                     crop_size=args.crop_size)

    print(f"Video clips: {n_total} total → "
          f"train={len(train_idx)} (vp {sorted(_TRAIN_VPS)}), "
          f"val={len(val_idx)}, test={len(test_idx)} "
          f"(vp {sorted(_VALTEST_VPS)})")
    return train_set, val_set, test_set, full_dataset.activity_labels, full_dataset.num_classes


class AugmentedVideoSubset(torch.utils.data.Dataset):
    """Re-load clips with augmentation=True, reusing the underlying dataset mapping."""

    def __init__(self, subset, parent_dataset, num_frames, crop_size):
        self.subset = subset
        # Create an augmented view sharing the same samples / video_map
        self._aug_ds = VideoDataset.__new__(VideoDataset)
        self._aug_ds.video_dir  = parent_dataset.video_dir
        self._aug_ds.num_frames = num_frames
        self._aug_ds.crop_size  = crop_size
        self._aug_ds.augment    = True
        self._aug_ds.samples    = parent_dataset.samples
        self._aug_ds.video_map  = parent_dataset.video_map
        self._aug_ds.activity_labels  = parent_dataset.activity_labels
        self._aug_ds.activity_to_idx  = parent_dataset.activity_to_idx
        self._aug_ds.num_classes      = parent_dataset.num_classes

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        original_idx = self.subset.indices[idx]
        return self._aug_ds[original_idx]


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, dpm_weight=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if dpm_weight > 0 and hasattr(model, 'forward_train'):
            logits, dpm_loss = model.forward_train(clips)
            loss = criterion(logits, labels) + dpm_weight * dpm_loss
        else:
            logits = model(clips)
            loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(clips)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = len(all_labels)
    return {
        'loss':             total_loss / n,
        'accuracy':         accuracy_score(all_labels, all_preds),
        'precision_macro':  precision_score(all_labels, all_preds, average='macro',    zero_division=0),
        'recall_macro':     recall_score   (all_labels, all_preds, average='macro',    zero_division=0),
        'f1_macro':         f1_score       (all_labels, all_preds, average='macro',    zero_division=0),
        'f1_weighted':      f1_score       (all_labels, all_preds, average='weighted', zero_division=0),
    }, all_preds, all_labels


def print_metrics(metrics, prefix=''):
    print(f"  {prefix}Loss: {metrics['loss']:.4f} | "
          f"Acc: {metrics['accuracy']:.4f} | "
          f"F1(macro): {metrics['f1_macro']:.4f} | "
          f"F1(wt): {metrics['f1_weighted']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data
    train_set, val_set, test_set, activity_labels, num_classes = build_datasets(args)
    pin = device.type == 'cuda'
    nw  = args.num_workers if device.type == 'cuda' else 0
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin)

    # Model
    print(f"\nArchitecture : {args.arch.upper()}")
    print(f"Training {num_classes} classes: {activity_labels}")
    ModelClass = MODEL_REGISTRY[args.arch]
    model = ModelClass(
        num_classes=num_classes,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {total_params:,} total / {trainable_params:,} trainable\n")

    # Optimizer — AdamW works better for Transformer-based models
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    # Cosine LR schedule with warm-up (5 epochs)
    warmup_epochs = min(5, args.epochs // 6)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(warmup_epochs, 1)
        progress = (ep - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_f1, best_epoch, history = 0.0, 0, []
    best_ckpt = os.path.join(OUTPUT_DIR, f'best_{args.arch}.pth')
    last_ckpt = os.path.join(OUTPUT_DIR, f'last_{args.arch}.pth')

    print(f"{'='*70}")
    print(f"Training [{args.arch.upper()}]: {args.epochs} epochs, "
          f"lr={args.lr}, batch={args.batch_size}, T={args.num_frames}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.dpm_weight)
        val_metrics, _, _     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs}  ({elapsed:.0f}s, lr={lr_now:.2e})")
        print(f"  Train  Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print_metrics(val_metrics, prefix='Val    ')

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_metrics['loss'], 'val_acc': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
        })

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch  = epoch
            torch.save(model.state_dict(), best_ckpt)
            print(f"  >> New best saved (F1={best_val_f1:.4f})")
        print()

    torch.save(model.state_dict(), last_ckpt)
    with open(os.path.join(OUTPUT_DIR, f'history_{args.arch}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # -----------------------------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Final Test Evaluation  (best model: epoch {best_epoch})")
    print(f"{'='*70}\n")

    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print("Test Results:")
    print_metrics(test_metrics, prefix='Test   ')
    print(f"  Precision(macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall   (macro): {test_metrics['recall_macro']:.4f}")

    # Classification report
    present = sorted(set(test_labels))
    target_names = [activity_labels[i] for i in present]
    report = classification_report(test_labels, test_preds, labels=present,
                                   target_names=target_names, zero_division=0)
    print(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_classes)))
    short = [n[:8] for n in activity_labels]
    header = "         " + " ".join(f"{n:>8s}" for n in short)
    print("Confusion Matrix:")
    print(header)
    for i, row in enumerate(cm):
        print(f"{short[i]:>8s}  " + " ".join(f"{v:8d}" for v in row))

    # Save detailed results
    results = {
        'arch': args.arch, 'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'confusion_matrix': cm.tolist(),
        'activity_labels': activity_labels,
        'args': vars(args),
    }
    with open(os.path.join(OUTPUT_DIR, f'results_{args.arch}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
