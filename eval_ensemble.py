"""
eval_ensemble.py — Evaluates inference-time ensembling of the static classifier
and the frozen-backbone temporal model across the validation dataset.

Weights w in [0.0, 0.1, ..., 1.0] where:
  ensemble_probs = w * static_probs + (1 - w) * temporal_probs
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

import config
import config_temporal as cfg
from model import load_model
from model_temporal import load_temporal_model
from datasets_temporal import build_temporal_dataset


def run_ensemble_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. Load Models ──
    static_ckpt_path = config.CKPT_PATH
    temporal_ckpt_path = cfg.CKPT_PATH_FROZEN

    print(f"Loading Static Model from: {static_ckpt_path}")
    model_static, _ = load_model(static_ckpt_path, device=device)
    model_static.eval()

    print(f"Loading Temporal Model from: {temporal_ckpt_path}")
    model_temporal, _ = load_temporal_model(temporal_ckpt_path, device=device)
    model_temporal.eval()

    # ── 2. Load Validation Dataset ──
    print("\nLoading validation sequences (DFEW + FERV39K)...")
    val_ds = build_temporal_dataset("val")
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )
    n_samples = len(val_ds)
    print(f"Total val clips: {n_samples:,}\n")

    # ── 3. Run Inference on All Clips ──
    static_probs_list = []
    temporal_probs_list = []
    true_labels_list = []

    print("Running static and temporal inference on all validation clips...")
    with torch.no_grad():
        for seqs, soft_labels in tqdm(val_loader, desc="  inference", ncols=80):
            B, T, C, H, W = seqs.shape
            seqs = seqs.to(device)
            trues = soft_labels.argmax(dim=1).numpy()
            true_labels_list.extend(trues)

            # (A) Temporal model prediction (clip level softmax)
            temporal_logits = model_temporal(seqs)
            t_probs = F.softmax(temporal_logits, dim=-1).cpu().numpy()
            temporal_probs_list.append(t_probs)

            # (B) Static model prediction (frame-averaged softmax)
            flat_frames = seqs.view(B * T, C, H, W)
            static_logits = model_static(flat_frames)
            s_probs_frames = F.softmax(static_logits, dim=-1).view(B, T, cfg.NUM_CLASSES)
            s_probs = s_probs_frames.mean(dim=1).cpu().numpy()
            static_probs_list.append(s_probs)

    static_all = np.vstack(static_probs_list)       # (N, 7)
    temporal_all = np.vstack(temporal_probs_list)   # (N, 7)
    trues_all = np.array(true_labels_list)           # (N,)

    # ── 4. Evaluate Ensemble Weights w in [0.0, 1.0] ──
    weights = [round(w, 1) for w in np.arange(0.0, 1.05, 0.1)]
    results = []

    for w in weights:
        # ensemble_probs = w * static + (1 - w) * temporal
        ens_probs = w * static_all + (1.0 - w) * temporal_all
        preds = ens_probs.argmax(axis=1)

        overall_acc = float((preds == trues_all).mean())

        # per-class accuracies
        per_class_acc = {}
        for c_idx, emo in enumerate(cfg.EMOTIONS):
            mask = (trues_all == c_idx)
            c_total = mask.sum()
            c_acc = float((preds[mask] == c_idx).sum() / c_total) if c_total > 0 else 0.0
            per_class_acc[emo] = c_acc

        results.append({
            "w": w,
            "overall_acc": overall_acc,
            "per_class": per_class_acc,
            "preds": preds,
        })

    # Sort results by overall accuracy descending
    results_sorted = sorted(results, key=lambda x: x["overall_acc"], reverse=True)

    # ── 5. Print Results Table ──
    print("\n" + "=" * 92)
    print("ENSEMBLE GRID SEARCH (w * static + (1-w) * temporal) — SORTED BY OVERALL ACCURACY")
    print("=" * 92)
    header = f"{'w (static)':<12} {'Overall Acc':>12} | " + " ".join(f"{e[:4]:>8}" for e in cfg.EMOTIONS)
    print(header)
    print("-" * 92)

    for r in results_sorted:
        w_str = f"{r['w']:.1f}"
        if r['w'] == 0.0:
            w_str += " (temp only)"
        elif r['w'] == 1.0:
            w_str += " (stat only)"
        row = f"{w_str:<12} {r['overall_acc']:>12.4f} | " + " ".join(
            f"{r['per_class'][e]:>8.3f}" for e in cfg.EMOTIONS
        )
        print(row)
    print("=" * 92)

    # ── 6. Best Weight & Confusion Matrix ──
    best = results_sorted[0]
    best_w = best["w"]
    best_acc = best["overall_acc"]
    best_preds = best["preds"]

    print(f"\n★ BEST ENSEMBLE WEIGHT: w = {best_w:.1f} (Static {int(best_w*100)}% / Temporal {int((1-best_w)*100)}%)")
    print(f"★ Best Overall Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")

    # Confusion matrix computation
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    for t, p in zip(trues_all, best_preds):
        cm[t, p] += 1

    print("CONFUSION MATRIX FOR BEST WEIGHT:")
    print("-" * 75)
    header_cm = f"{'True \\ Pred':<12} " + " ".join(f"{e[:7]:>8}" for e in cfg.EMOTIONS) + f"{'Total':>8}"
    print(header_cm)
    print("-" * 75)
    for i, emo in enumerate(cfg.EMOTIONS):
        row_str = f"{emo:<12} " + " ".join(f"{cm[i, j]:>8d}" for j in range(cfg.NUM_CLASSES))
        row_str += f"{cm[i].sum():>8d}"
        print(row_str)
    print("-" * 75)
    print(f"{'Total Pred':<12} " + " ".join(f"{cm[:, j].sum():>8d}" for j in range(cfg.NUM_CLASSES)))
    print("=" * 75)


if __name__ == "__main__":
    run_ensemble_eval()
