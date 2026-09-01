"""
eval_confusion.py — Run any checkpoint on the val set and print a real confusion matrix.

Unchanged from before.

Usage:
    python eval_confusion.py                                        # uses best_model.pt
    python eval_confusion.py --ckpt checkpoints/some_other.pt
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

import config
from datasets import build_dataset
from model import load_model

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("  matplotlib not installed — will print text matrix only")


def run_eval(ckpt_path: Path, batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
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

    print("Loading val dataset...")
    val_ds = build_dataset("val")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=(device.type == "cuda"))

    num_classes = config.NUM_CLASSES
    emotions    = config.EMOTIONS

    conf_matrix  = np.zeros((num_classes, num_classes), dtype=np.int64)
    class_scores = defaultdict(list)

    print("\nRunning inference on val set...\n")
    model.eval()
    with torch.no_grad():
        for imgs, soft_labels in tqdm(val_loader, desc="  eval", ncols=80):
            imgs        = imgs.to(device)
            soft_labels = soft_labels.to(device)
            logits      = model(imgs)
            probs       = F.softmax(logits, dim=1)
            preds       = probs.argmax(dim=1).cpu().numpy()
            trues       = soft_labels.argmax(dim=1).cpu().numpy()
            confs       = probs.max(dim=1).values.cpu().numpy()
            for p, t, c in zip(preds, trues, confs):
                conf_matrix[t, p] += 1
                class_scores[t].append(c)

    print("=" * 62)
    print(f"{'Class':<12} {'Correct':>8} {'Total':>8} {'Acc':>8} {'Avg conf':>10}")
    print("-" * 62)
    per_class_acc = {}
    for i, emo in enumerate(emotions):
        total   = conf_matrix[i].sum()
        correct = conf_matrix[i, i]
        acc     = correct / max(total, 1)
        avg_c   = float(np.mean(class_scores[i])) if class_scores[i] else 0.0
        per_class_acc[emo] = acc
        print(f"{emo:<12} {correct:>8d} {total:>8d} {acc:>8.3f} {avg_c:>10.3f}")
    print("=" * 62)
    overall_acc = conf_matrix.diagonal().sum() / conf_matrix.sum()
    print(f"{'Overall':<12} {'':>8} {'':>8} {overall_acc:>8.3f}\n")

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
        print(f"  {true_e:<10} → predicted as {pred_e:<10}  {rate:.1%}  ({count} samples)")

    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        norm = conf_matrix.astype(float)
        norm = norm / np.maximum(norm.sum(axis=1, keepdims=True), 1)

        ax = axes[0]
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(emotions, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(emotions, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"Confusion matrix — {ckpt_path.stem}", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(num_classes):
            for j in range(num_classes):
                val = norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.55 else "black")

        ax2 = axes[1]
        accs   = [per_class_acc[e] for e in emotions]
        colors = ["#2ecc71" if a >= 0.70 else "#f39c12" if a >= 0.60 else "#e74c3c" for a in accs]
        bars   = ax2.barh(emotions, accs, color=colors, edgecolor="white", linewidth=0.5)
        ax2.axvline(0.70, color="#2c3e50", linestyle="--", linewidth=1, alpha=0.6)
        ax2.set_xlim(0, 1.0)
        ax2.set_xlabel("Accuracy", fontsize=11)
        ax2.set_title("Per-class accuracy", fontsize=12, fontweight="bold")
        for bar, acc in zip(bars, accs):
            ax2.text(acc + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{acc:.1%}", va="center", fontsize=9)
        legend_patches = [
            mpatches.Patch(color="#2ecc71", label=">=70% good"),
            mpatches.Patch(color="#f39c12", label="60-70% needs work"),
            mpatches.Patch(color="#e74c3c", label="<60% weak"),
        ]
        ax2.legend(handles=legend_patches, fontsize=9, loc="lower right")

        plt.tight_layout()
        out_path = config.CKPT_DIR / f"confusion_{ckpt_path.stem}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved → {out_path}")

    return conf_matrix, per_class_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(config.CKPT_PATH),
                        help="Path to checkpoint (default: best_model.pt)")
    args = parser.parse_args()
    run_eval(Path(args.ckpt))
