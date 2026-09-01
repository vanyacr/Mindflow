"""
eval_confusion_static.py — same confusion matrix / per-class breakdown as
eval_confusion.py, but restricted to ONLY the 6 static sources (affectnet,
ckplus, fer_ck48, fer_k7, fer_stock, rafdb) — i.e. the exact same val
population your pre-retrain 74.3% baseline was measured on.

This is the number to compare against your old confusion matrix, not the
blended 8-source one from eval_confusion.py.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

import config
from model import EmotionModel
from datasets import (
    AffectNetDataset, CKPlusDataset, FERCk48Dataset, FERKaggle7Dataset,
    FERStockDataset, RAFDBDataset, get_transforms,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def build_static_val():
    tf = get_transforms("val")
    parts = [
        AffectNetDataset("val", tf),
        CKPlusDataset("val", tf),
        FERCk48Dataset("val", tf),
        FERKaggle7Dataset("val", tf),
        FERStockDataset("val", tf),
        RAFDBDataset("val", tf),
    ]
    parts = [p for p in parts if len(p) > 0]
    combined = ConcatDataset(parts)
    print(f"\n  Total static val: {len(combined)} samples from {len(parts)} datasets\n")
    return combined


def main(ckpt_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(ckpt_path or config.CKPT_PATH)
    print(f"Checkpoint: {ckpt_path}\n")

    state = torch.load(str(ckpt_path), map_location=device)
    if "head.0.weight" in state and state["head.0.weight"].shape[1] == 1792:
        from model_b4 import EmotionModelB4
        model = EmotionModelB4().to(device)
        model.load_state_dict(state)
        print("  Detected EfficientNet-B4 architecture from checkpoint")
    else:
        model = EmotionModel().to(device)
        model.load_state_dict(state)
        print("  Detected EfficientNet-B2 architecture from checkpoint")
    model.eval()

    val_ds = build_static_val()
    val_loader = DataLoader(val_ds, batch_size=96, shuffle=False,
                             num_workers=4, pin_memory=(device.type == "cuda"))

    num_classes = config.NUM_CLASSES
    emotions = config.EMOTIONS
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    class_scores = defaultdict(list)

    print("Running inference on STATIC-ONLY val set...\n")
    with torch.no_grad():
        for imgs, soft_labels in tqdm(val_loader, desc="  eval", ncols=80):
            imgs = imgs.to(device)
            soft_labels = soft_labels.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            trues = soft_labels.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()
            for p, t, c in zip(preds, trues, confs):
                conf_matrix[t, p] += 1
                class_scores[t].append(c)

    print("=" * 62)
    print(f"{'Class':<12} {'Correct':>8} {'Total':>8} {'Acc':>8} {'Avg conf':>10}")
    print("-" * 62)
    per_class_acc = {}
    for i, emo in enumerate(emotions):
        total = conf_matrix[i].sum()
        correct = conf_matrix[i, i]
        acc = correct / max(total, 1)
        avg_c = float(np.mean(class_scores[i])) if class_scores[i] else 0.0
        per_class_acc[emo] = acc
        print(f"{emo:<12} {correct:>8d} {total:>8d} {acc:>8.3f} {avg_c:>10.3f}")
    print("=" * 62)
    overall_acc = conf_matrix.diagonal().sum() / conf_matrix.sum()
    print(f"{'Overall (static only)':<12} {'':>8} {'':>8} {overall_acc:>8.3f}\n")

    print("\nConfusion matrix (rows = true, cols = predicted):\n")
    header = f"{'':>10}" + "".join(f"{e[:4]:>7}" for e in emotions)
    print(header)
    print("-" * (10 + 7 * num_classes))
    for i, emo in enumerate(emotions):
        row = f"{emo[:10]:<10}" + "".join(f"{conf_matrix[i,j]:>7d}" for j in range(num_classes))
        print(row)

    print("\nTop confused pairs (excluding diagonal):\n")
    off_diag = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix[i, j] > 0:
                rate = conf_matrix[i, j] / max(conf_matrix[i].sum(), 1)
                off_diag.append((rate, emotions[i], emotions[j], conf_matrix[i, j]))
    off_diag.sort(reverse=True)
    for rate, true_e, pred_e, count in off_diag[:10]:
        print(f"  {true_e:<10} -> predicted as {pred_e:<10}  {rate:.1%}  ({count} samples)")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 6))
        norm = conf_matrix.astype(float)
        norm = norm / np.maximum(norm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(emotions, rotation=45, ha="right")
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(emotions)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion matrix — static val only (apples-to-apples baseline)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(num_classes):
            for j in range(num_classes):
                val = norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.55 else "black")
        plt.tight_layout()
        out_path = config.CKPT_DIR / "confusion_static_only.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(config.CKPT_PATH),
                        help="Path to checkpoint (default: best_model.pt)")
    args = parser.parse_args()
    main(args.ckpt)
