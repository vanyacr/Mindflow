"""
eval_video_static_b4.py — Evaluates the EfficientNet-B4 static model on the 10,188 video
validation clips (DFEW + FERV39K), using per-frame classification and softmax averaging
over the clip's 16 frames.

Directly comparable against the 50.76% baseline achieved by EfficientNet-B2 in eval_temporal_vs_static.py.
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
from datasets_temporal import build_temporal_dataset
from model_b4 import load_model_b4


def run_video_eval(ckpt_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ckpt_path = Path(ckpt_path or (config.CKPT_DIR / "best_model_b4.pt"))
    print(f"Loading B4 Static Model from: {ckpt_path}")
    model_b4, _ = load_model_b4(ckpt_path, device=device)
    model_b4.eval()

    print("\nLoading video validation sequences (DFEW + FERV39K)...")
    val_ds = build_temporal_dataset("val")
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )
    n_samples = len(val_ds)
    print(f"Total video val clips: {n_samples:,}\n")

    source_per_sample = []
    for ds in val_ds.datasets:
        source_per_sample.extend([ds.dataset_name] * len(ds))

    n_classes = cfg.NUM_CLASSES
    conf_static_b4 = np.zeros((n_classes, n_classes), dtype=np.int64)
    source_correct = defaultdict(int)
    source_total = defaultdict(int)

    sample_idx = 0
    print("Running B4 static frame-averaged inference on video val clips...\n")

    with torch.no_grad():
        for seqs, soft_labels in tqdm(val_loader, desc="  eval B4", ncols=80):
            B, T, C, H, W = seqs.shape
            seqs = seqs.to(device)
            trues = soft_labels.argmax(dim=1).numpy()

            # Static baseline: per-frame classify + average softmax across 16 frames
            flat = seqs.view(B * T, C, H, W)
            static_logits = model_b4(flat)
            static_probs = F.softmax(static_logits, dim=1).view(B, T, n_classes)
            avg_probs = static_probs.mean(dim=1)
            preds = avg_probs.argmax(dim=1).cpu().numpy()

            for i in range(B):
                t, p = trues[i], preds[i]
                conf_static_b4[t, p] += 1
                src = source_per_sample[sample_idx]
                source_total[src] += 1
                if p == t:
                    source_correct[src] += 1
                sample_idx += 1

    overall_acc = conf_static_b4.diagonal().sum() / conf_static_b4.sum()

    print("\n" + "=" * 64)
    print(f"{'Model / Pipeline':<36} {'Overall Video Acc':>20}")
    print("-" * 64)
    print(f"{'EfficientNet-B4 Static Avg':<36} {overall_acc:>20.4f}")
    print(f"{'EfficientNet-B2 Static Avg (Baseline)':<36} {'0.5076':>20}")
    print("=" * 64)

    delta = overall_acc - 0.5076
    sign = "+" if delta >= 0 else ""
    print(f"\nB4 vs B2 Delta on Video Val: {sign}{delta:.4f} ({sign}{delta*100:.2f} pp)")

    # Per-class accuracy
    print(f"\n{'Class':<12} {'Correct':>8} {'Total':>8} {'Acc (B4)':>10} {'B2 Baseline':>12}")
    print("-" * 54)
    b2_per_class = {
        "happy": 0.654, "sad": 0.545, "angry": 0.473, "neutral": 0.608,
        "fear": 0.190, "disgust": 0.264, "surprise": 0.268
    }
    for i, emo in enumerate(cfg.EMOTIONS):
        total = conf_static_b4[i].sum()
        correct = conf_static_b4[i, i]
        acc = correct / max(total, 1)
        b2_val = b2_per_class.get(emo, 0.0)
        d = acc - b2_val
        dsign = "+" if d >= 0 else ""
        print(f"{emo:<12} {correct:>8d} {total:>8d} {acc:>10.3f} {b2_val:>10.3f} ({dsign}{d:.3f})")
    print("=" * 54)

    # Per-source breakdown
    print(f"\n{'Source':<12} {'B4 Video Acc':>14} {'N clips':>10}")
    print("-" * 40)
    for src in sorted(source_total):
        acc = source_correct[src] / max(source_total[src], 1)
        print(f"{src:<12} {acc:>14.3f} {source_total[src]:>10}")
    print("=" * 40)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to B4 checkpoint (default: checkpoints/best_model_b4.pt)")
    args = parser.parse_args()
    run_video_eval(args.ckpt)
