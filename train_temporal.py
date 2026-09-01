"""
train_temporal.py — Track B (BiGRU) training, rebuilt from scratch.

Fixes applied vs the paused run that overfit:
  - Class weight cap at 4.0x (config_temporal.CLASS_WEIGHT_CAP) — was 11.5x
    uncapped on disgust (116 DFEW training samples)
  - Label smoothing 0.10 — softens hard-label overfitting on tiny classes
  - Lower LR (3e-4) + higher weight decay (5e-4) — appropriate for training
    only a BiGRU + head on frozen embeddings, not a full network
  - Frozen EfficientNet-B2 backbone (already validated at 74.6% static acc)
    means far fewer trainable params than a from-scratch temporal CNN

Run:
    python train_temporal.py

Checkpoint saved to:  checkpoints/best_model_temporal.pt
Training log saved to: checkpoints/train_temporal_log.csv
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.stdout.reconfigure(encoding='utf-8')

import config_temporal as cfg
from datasets_temporal import build_temporal_dataset, build_temporal_weighted_sampler
from model_temporal import TemporalEmotionModel, TemporalTransformerEmotionModel

BACKBONE_LR = 1e-5


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing: float = cfg.LABEL_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, soft_targets):
        n_classes = logits.size(1)
        smooth = soft_targets * (1 - self.smoothing) + self.smoothing / n_classes
        log_probs = F.log_softmax(logits, dim=1)
        return -(smooth * log_probs).sum(dim=1).mean()


class FocalSoftCrossEntropyLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for multi-class video emotion recognition.
    Downweights easy/dominant classes (e.g. happy/neutral) and forces gradient focus
    on hard, subtle, and underrepresented expressions (fear, disgust, surprise, sad).
    """
    def __init__(self, gamma: float = 1.5, smoothing: float = cfg.LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, soft_targets):
        n_classes = logits.size(1)
        smooth = soft_targets * (1 - self.smoothing) + self.smoothing / n_classes
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        focal_weight = torch.pow(1.0 - probs, self.gamma)
        return -(smooth * focal_weight * log_probs).sum(dim=1).mean()


def compute_accuracy(logits, soft_targets):
    pred = logits.argmax(dim=1)
    true = soft_targets.argmax(dim=1)
    return (pred == true).float().mean().item()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()   # backbone stays eval() via TemporalEmotionModel / TemporalTransformerEmotionModel train() override
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for seqs, soft_labels in tqdm(loader, desc="  train", leave=False, ncols=80):
        seqs = seqs.to(device)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, soft_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=2.0
        )
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(logits, soft_labels)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    class_correct = np.zeros(cfg.NUM_CLASSES)
    class_total = np.zeros(cfg.NUM_CLASSES)

    with torch.no_grad():
        for seqs, soft_labels in tqdm(loader, desc="  val  ", leave=False, ncols=80):
            seqs = seqs.to(device)
            soft_labels = soft_labels.to(device)

            logits = model(seqs)
            loss = criterion(logits, soft_labels)

            total_loss += loss.item()
            total_acc += compute_accuracy(logits, soft_labels)
            n_batches += 1

            preds = logits.argmax(1).cpu().numpy()
            trues = soft_labels.argmax(1).cpu().numpy()
            for p, t in zip(preds, trues):
                class_total[t] += 1
                class_correct[t] += int(p == t)

    per_class = {
        cfg.IDX_TO_EMOTION[i]: round(class_correct[i] / max(class_total[i], 1), 3)
        for i in range(cfg.NUM_CLASSES)
    }
    return total_loss / n_batches, total_acc / n_batches, per_class


def train(arch: str = "transformer", use_focal: bool = True, unfreeze_backbone: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print("Loading training sequences...")
    train_ds = build_temporal_dataset("train")
    print("Loading validation sequences...")
    val_ds = build_temporal_dataset("val")

    sampler = build_temporal_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    is_transformer = (arch == "transformer")
    if is_transformer:
        print("Architecture: Spatial-Temporal Transformer (Former-DFER style, 1408->512 features)")
        model = TemporalTransformerEmotionModel().to(device)
        save_ckpt_path = cfg.CKPT_PATH_TRANSFORMER
        log_path = cfg.CKPT_DIR / "train_temporal_transformer_log.csv"
    else:
        print("Architecture: BiGRU with Attention Pooling")
        model = TemporalEmotionModel().to(device)
        if unfreeze_backbone:
            model.unfreeze_last_block()
            save_ckpt_path = cfg.CKPT_PATH_UNFROZEN
            log_path = cfg.CKPT_DIR / "train_temporal_unfrozen_log.csv"
        else:
            save_ckpt_path = cfg.CKPT_PATH_FROZEN
            log_path = cfg.CKPT_DIR / "train_temporal_frozen_log.csv"

    model.static_model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")

    if use_focal:
        criterion = FocalSoftCrossEntropyLoss(gamma=1.5, smoothing=cfg.LABEL_SMOOTHING)
        print("Loss function: Focal Loss (gamma=1.5) + Label Smoothing (0.10)")
    else:
        criterion = SoftCrossEntropyLoss(smoothing=cfg.LABEL_SMOOTHING)
        print("Loss function: Soft Cross-Entropy (smoothing=0.10)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )

    best_val_acc = 0.0
    log_rows = []
    patience = 0
    MAX_PATIENCE = 10

    print(f"\nStarting training — {cfg.EPOCHS} epochs")
    print(f"Weight cap: {cfg.CLASS_WEIGHT_CAP}x | LR: {cfg.LR} | WD: {cfg.WEIGHT_DECAY}\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("─" * 64)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if is_transformer:
                ckpt = {
                    "proj": model.proj.state_dict(),
                    "pos_embed": model.pos_embed.data,
                    "transformer": model.transformer.state_dict(),
                    "attn": model.attn.state_dict(),
                    "head": model.head.state_dict(),
                    "residual_scale": model.residual_scale.data,
                }
            else:
                ckpt = {
                    "gru": model.gru.state_dict(),
                    "attn": model.attn.state_dict(),
                    "head": model.head.state_dict(),
                    "residual_scale": model.residual_scale.data,
                }
                if unfreeze_backbone:
                    ckpt["backbone_last_block"] = model.static_model.backbone.blocks[-1].state_dict()
            torch.save(ckpt, str(save_ckpt_path))
            tag = "  ✓ saved"
            patience = 0
        else:
            patience += 1

        scale_val = model.residual_scale.item()
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>7.4f}  {current_lr:>8.2e}  "
              f"(scale={scale_val:.3f}){tag}")

        if epoch % 5 == 0:
            pc_str = "  ".join(f"{e}:{v:.2f}" for e, v in per_class.items())
            print(f"         per-class: {pc_str}")
            weak = [e for e, v in per_class.items() if v < 0.40]
            if weak:
                print(f"         ⚠ still weak: {weak}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "residual_scale": round(scale_val, 4),
            "lr": current_lr,
            **{f"acc_{e}": per_class[e] for e in cfg.EMOTIONS},
        })

        gap = train_acc - val_acc
        if gap > 0.20:
            print(f"         ⚠ train/val gap = {gap:.3f} — watch for overfitting")

        if patience >= MAX_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {MAX_PATIENCE} epochs)")
            break

    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print(f"\n{'─'*64}")
    print("Training complete.")
    print(f"Best val accuracy : {best_val_acc:.4f}")
    print(f"Checkpoint        : {save_ckpt_path}")
    print(f"Training log      : {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="transformer", choices=["transformer", "bigru"],
                        help="Temporal architecture: 'transformer' (default) or 'bigru'")
    parser.add_argument("--no_focal", action="store_true", help="Use standard Soft-CE instead of Focal Loss")
    parser.add_argument("--unfreeze_backbone", action="store_true",
                        help="Unfreeze backbone last block (only for bigru)")
    args = parser.parse_args()
    train(arch=args.arch, use_focal=not args.no_focal, unfreeze_backbone=args.unfreeze_backbone)
