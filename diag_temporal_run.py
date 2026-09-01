"""
diag_temporal_run.py — Diagnostic script to evaluate the attention-pooled Track B checkpoint.

Checks:
1. Checkpoint file integrity and structure (gru, attn, head).
2. Val set accuracy overall and per-class (specifically neutral and sad).
3. Confusion matrix to see what neutral and sad are misclassified as.
4. Per-source breakdown (DFEW vs FERV39K).
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import config_temporal as cfg
from datasets_temporal import build_temporal_dataset
from model_temporal import TemporalEmotionModel


sys.stdout.reconfigure(encoding='utf-8')

def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 70)

    ckpt_path = cfg.CKPT_PATH
    print(f"Inspecting checkpoint: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    ckpt = torch.load(str(ckpt_path), map_location=device)
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"  [{k}] sub-keys: {list(v.keys())}")
        else:
            print(f"  [{k}]: {type(v)}")

    # Build model and load state dict
    print("\nInitializing TemporalEmotionModel...")
    model = TemporalEmotionModel().to(device)
    model.static_model.to(device)

    model.gru.load_state_dict(ckpt["gru"])
    if "attn" in ckpt:
        model.attn.load_state_dict(ckpt["attn"])
        print("  [OK] Loaded attention pooling weights successfully")
    else:
        print("  [WARNING] 'attn' key not found in checkpoint!")

    model.head.load_state_dict(ckpt["head"])
    if "backbone_last_block" in ckpt:
        model.static_model.backbone.blocks[-1].load_state_dict(ckpt["backbone_last_block"])
        print("  [OK] Loaded domain-adapted backbone last block")

    model.eval()

    print("\nLoading validation dataset...")
    val_ds = build_temporal_dataset("val")
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda")
    )

    source_per_sample = []
    for ds in val_ds.datasets:
        source_per_sample.extend([ds.dataset_name] * len(ds))

    n_classes = cfg.NUM_CLASSES
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    source_correct = defaultdict(int)
    source_total = defaultdict(int)

    sample_idx = 0
    print("\nRunning evaluation on validation set...")
    with torch.no_grad():
        for seqs, soft_labels in tqdm(val_loader, desc="  eval", ncols=80):
            B = seqs.shape[0]
            seqs = seqs.to(device)
            trues = soft_labels.argmax(dim=1).numpy()

            logits = model(seqs)
            preds = logits.argmax(dim=1).cpu().numpy()

            for i in range(B):
                t, p = trues[i], preds[i]
                conf[t, p] += 1
                src = source_per_sample[sample_idx]
                source_total[src] += 1
                if p == t:
                    source_correct[src] += 1
                sample_idx += 1

    overall_acc = conf.diagonal().sum() / conf.sum()

    print("\n" + "=" * 70)
    print(f"OVERALL VALIDATION ACCURACY: {overall_acc * 100:.2f}% ({conf.diagonal().sum()} / {conf.sum()})")
    print("=" * 70)

    print("\nPER-CLASS ACCURACY BREAKDOWN:")
    print(f"{'Class':<12} {'Correct':>10} {'Total':>10} {'Accuracy':>10}")
    print("-" * 46)
    per_class_acc = {}
    for i, emo in enumerate(cfg.EMOTIONS):
        corr = conf[i, i]
        tot = max(conf[i].sum(), 1)
        acc = corr / tot
        per_class_acc[emo] = acc
        print(f"{emo:<12} {corr:>10d} {tot:>10d} {acc * 100:>9.2f}%")

    print("\n" + "=" * 70)
    print("PER-SOURCE BREAKDOWN:")
    print(f"{'Source':<12} {'Correct':>10} {'Total':>10} {'Accuracy':>10}")
    print("-" * 46)
    for src in sorted(source_total):
        s_corr = source_correct[src]
        s_tot = source_total[src]
        s_acc = s_corr / max(s_tot, 1)
        print(f"{src:<12} {s_corr:>10d} {s_tot:>10d} {s_acc * 100:>9.2f}%")

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (Rows = True, Columns = Pred)")
    print("=" * 70)
    header = f"{'':>10}" + "".join(f"{e[:4]:>7}" for e in cfg.EMOTIONS)
    print(header)
    print("-" * (10 + 7 * n_classes))
    for i, emo in enumerate(cfg.EMOTIONS):
        row = f"{emo[:10]:<10}" + "".join(f"{conf[i,j]:>7d}" for j in range(n_classes))
        print(row)

    print("\n" + "=" * 70)
    print("WHERE DO TRUE 'NEUTRAL' AND 'SAD' CLIPS END UP?")
    print("=" * 70)
    for target in ["neutral", "sad"]:
        idx = cfg.EMOTION_TO_IDX[target]
        row = conf[idx]
        tot = max(row.sum(), 1)
        print(f"\nTrue '{target}' clips (total {tot}):")
        for j, emo in enumerate(cfg.EMOTIONS):
            pct = 100 * row[j] / tot
            mark = "  <-- RECALL" if emo == target else ""
            print(f"  -> predicted {emo:<10} {row[j]:>6d} ({pct:5.2f}%){mark}")

    print("\nDone.")


if __name__ == "__main__":
    run_diagnostics()
