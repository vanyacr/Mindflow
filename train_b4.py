"""
train_b4.py — Trains EmotionModelB4 (EfficientNet-B4 backbone) from scratch
on the exact same static dataset mix (AffectNet, CK+, FER+, RAF-DB, DFEW frame-samples,
FERV39K frame-samples).

Checkpoints saved to:  checkpoints/best_model_b4.pt
Training log saved to: checkpoints/train_log_b4.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

import config
from datasets import build_dataset, build_weighted_sampler
from model_b4 import EmotionModelB4

CKPT_PATH_B4 = config.CKPT_DIR / "best_model_b4.pt"
LOG_PATH_B4  = config.CKPT_DIR / "train_log_b4.csv"
BATCH_SIZE   = 64   # Adjusted from 96 to 64 for B4 memory safety and stable gradients


# ─────────────────────────────────────────────────────────────
#  LOSS
# ─────────────────────────────────────────────────────────────

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, soft_targets):
        n_classes = logits.size(1)
        smooth = soft_targets * (1 - self.smoothing) + self.smoothing / n_classes
        log_probs = F.log_softmax(logits, dim=1)
        return -(smooth * log_probs).sum(dim=1).mean()


# ─────────────────────────────────────────────────────────────
#  MIXUP
# ─────────────────────────────────────────────────────────────

def mixup_batch(imgs, labels, alpha: float = 0.3):
    if np.random.random() > 0.5:
        return imgs, labels

    lam = np.random.beta(alpha, alpha)
    batch_size = imgs.size(0)
    idx = torch.randperm(batch_size, device=imgs.device)

    mixed_imgs   = lam * imgs   + (1 - lam) * imgs[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_imgs, mixed_labels


# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────

def compute_accuracy(logits, soft_targets):
    pred = logits.argmax(dim=1)
    true = soft_targets.argmax(dim=1)
    return (pred == true).float().mean().item()


# ─────────────────────────────────────────────────────────────
#  TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0
    use_amp    = (device.type == "cuda")

    for imgs, soft_labels in tqdm(loader, desc="  train", leave=False, ncols=80):
        imgs        = imgs.to(device)
        soft_labels = soft_labels.to(device)

        imgs, soft_labels = mixup_batch(imgs, soft_labels, alpha=0.3)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, soft_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc  += compute_accuracy(logits, soft_labels)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ─────────────────────────────────────────────────────────────
#  VALIDATE
# ─────────────────────────────────────────────────────────────

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0
    use_amp    = (device.type == "cuda")

    class_correct = np.zeros(config.NUM_CLASSES)
    class_total   = np.zeros(config.NUM_CLASSES)

    with torch.no_grad():
        for imgs, soft_labels in tqdm(loader, desc="  val  ", leave=False, ncols=80):
            imgs        = imgs.to(device)
            soft_labels = soft_labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, soft_labels)

            total_loss += loss.item()
            total_acc  += compute_accuracy(logits, soft_labels)
            n_batches  += 1

            preds = logits.argmax(1).cpu().numpy()
            trues = soft_labels.argmax(1).cpu().numpy()
            for p, t in zip(preds, trues):
                class_total[t]   += 1
                class_correct[t] += int(p == t)

    per_class = {
        config.IDX_TO_EMOTION[i]: round(class_correct[i] / max(class_total[i], 1), 3)
        for i in range(config.NUM_CLASSES)
    }
    return total_loss / n_batches, total_acc / n_batches, per_class


# ─────────────────────────────────────────────────────────────
#  MAIN TRAINING
# ─────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("AMP: ON (mixed precision)")
        print(f"Model: EfficientNet-B4 | Batch Size: {BATCH_SIZE}\n")

    print("Loading training datasets...")
    train_ds = build_dataset("train")
    print("Loading validation datasets...")
    val_ds   = build_dataset("val")

    sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = EmotionModelB4().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} (vs 8.9M for B2)")

    criterion = SoftCrossEntropyLoss(smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=5e-7
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    log_rows     = []
    patience     = 0
    MAX_PATIENCE = 15

    print(f"\nStarting training — {config.EPOCHS} epochs\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("─" * 64)

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, per_class = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(CKPT_PATH_B4))
            tag = "  ✓ saved"
            patience = 0
        else:
            patience += 1

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>7.4f}  {current_lr:>8.2e}{tag}")

        if epoch % 5 == 0:
            pc_str = "  ".join(f"{e}:{v:.2f}" for e, v in per_class.items())
            print(f"         per-class: {pc_str}")
            weak = [e for e, v in per_class.items() if v < 0.62]
            if weak:
                print(f"         ⚠ still weak: {weak}")

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc, 5),
            "val_loss":   round(val_loss, 5),
            "val_acc":    round(val_acc, 5),
            "lr":         current_lr,
            **{f"acc_{e}": per_class[e] for e in config.EMOTIONS},
        })

        if patience >= MAX_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {MAX_PATIENCE} epochs)")
            break

    pd.DataFrame(log_rows).to_csv(LOG_PATH_B4, index=False)

    print(f"\n{'─'*64}")
    print("Training complete.")
    print(f"Best val accuracy : {best_val_acc:.4f}")
    print(f"Checkpoint        : {CKPT_PATH_B4}")
    print(f"Training log      : {LOG_PATH_B4}")


if __name__ == "__main__":
    train()
