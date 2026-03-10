"""
ST-GCN Training & Evaluation for Skeleton-based Action Recognition.

Usage:
    python train.py                          # train with default settings
    python train.py --epochs 100 --lr 0.01   # custom hyperparameters
    python train.py --split 0                # use specific data split
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from stgcn.model import STGCN
from stgcn.dataset import PoseDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_DIR = os.path.join(BASE_DIR, 'pose_all')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def parse_args():
    parser = argparse.ArgumentParser(description='ST-GCN Skeleton Action Recognition')
    parser.add_argument('--label_csv', type=str,
                        default=os.path.join(BASE_DIR, 'activity_label', 'tasklevel.chunks_90.csv'),
                        help='Path to activity label CSV file')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                        help='Subset of class names to train on (default: all classes in label file)')
    parser.add_argument('--split', type=int, default=0, choices=[0, 1, 2],
                        help='Data split index (default: 0)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--max_frames', type=int, default=90,
                        help='Temporal window size')
    parser.add_argument('--graph_strategy', type=str, default='spatial',
                        choices=['uniform', 'distance', 'spatial'])
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader num_workers (set to 0 on Windows)')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def build_datasets(args):
    """
    Build train/val/test datasets.

    Uses --label_csv and optionally filters to --classes subset.
    Performs a 70/15/15 random split among the matched samples.
    """
    full_dataset = PoseDataset(
        args.label_csv, POSE_DIR,
        max_frames=args.max_frames, augment=False,
        allowed_classes=args.classes,
    )
    n_total = len(full_dataset)

    if n_total == 0:
        raise RuntimeError("No samples found. Check --label_csv and --classes.")

    # Split: 70% train, 15% val, 15% test
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Create augmented wrapper for training
    train_dataset = AugmentedSubset(train_set, augment=True)

    print(f"Dataset: {n_total} samples -> train={n_train}, val={n_val}, test={n_test}")
    return train_dataset, val_set, test_set, full_dataset.activity_labels, full_dataset.num_classes


class AugmentedSubset(torch.utils.data.Dataset):
    """Wrapper that enables augmentation for a Subset."""

    def __init__(self, subset, augment=True):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        skeleton, label = self.subset[idx]
        if self.augment:
            skeleton = self._augment(skeleton)
        return skeleton, label

    def _augment(self, skeleton):
        """skeleton: (C=3, T, V) tensor"""
        s = skeleton.numpy()
        # Random noise
        if np.random.rand() < 0.5:
            s = s + np.random.randn(*s.shape).astype(np.float32) * 0.01
        # Random scale
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            s = s * scale
        return torch.from_numpy(s.copy())


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for skeletons, labels in dataloader:
        skeletons = skeletons.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(skeletons)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for skeletons, labels in dataloader:
        skeletons = skeletons.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(skeletons)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = len(all_labels)

    metrics = {
        'loss': total_loss / n,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
    }
    return metrics, all_preds, all_labels


def print_metrics(metrics, prefix=''):
    print(f"  {prefix}Loss: {metrics['loss']:.4f} | "
          f"Acc: {metrics['accuracy']:.4f} | "
          f"F1(macro): {metrics['f1_macro']:.4f} | "
          f"F1(weighted): {metrics['f1_weighted']:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Build datasets
    train_set, val_set, test_set, activity_labels, num_classes = build_datasets(args)

    pin = (device.type == 'cuda')
    nw = args.num_workers if device.type == 'cuda' else 0
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=pin)

    print(f"Training {num_classes} classes: {activity_labels}")

    # Build model (official ST-GCN 9-layer architecture)
    model = STGCN(
        num_classes=num_classes,
        in_channels=3,
        graph_strategy=args.graph_strategy,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Loss & optimizer (official ST-GCN uses SGD + Nesterov)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 50, 70], gamma=0.1
    )

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    history = []

    print(f"\n{'='*70}")
    print(f"Starting training: {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}")
    print(f"Graph strategy: {args.graph_strategy}, dropout: {args.dropout}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s, lr={lr_now:.6f})")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print_metrics(val_metrics, prefix='Val   ')

        epoch_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
        }
        history.append(epoch_info)

        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  >> New best model saved (F1={best_val_f1:.4f})")

        print()

    # Save last model and training history
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'last_model.pth'))
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # =========================================================================
    # Final Evaluation on Test Set
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Final Evaluation (best model from epoch {best_epoch})")
    print(f"{'='*70}\n")

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'),
                                     map_location=device, weights_only=True))
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print("Test Set Results:")
    print_metrics(test_metrics, prefix='Test  ')
    print(f"  Precision(macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall(macro):    {test_metrics['recall_macro']:.4f}")

    # Per-class report
    present_labels = sorted(set(test_labels))
    target_names = [activity_labels[i] for i in present_labels]
    report = classification_report(
        test_labels, test_preds,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
    )
    print(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_classes)))
    print("Confusion Matrix:")
    short_names = [n[:8] for n in activity_labels]
    header = "         " + " ".join(f"{n:>8s}" for n in short_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:8d}" for v in row)
        print(f"{short_names[i]:>8s}  {row_str}")

    # Save results
    results = {
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'confusion_matrix': cm.tolist(),
        'activity_labels': activity_labels,
        'args': vars(args),
    }
    with open(os.path.join(OUTPUT_DIR, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
