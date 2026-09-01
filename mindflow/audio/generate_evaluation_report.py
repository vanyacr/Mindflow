"""
MindFlow — Evaluation Report Generator.

Produces report-ready artifacts for Phase 13 (Evaluation):
    1. stage1_confusion_matrix.png   -- heatmap, emotion classification
    2. stage1_classification_report.txt -- precision/recall/f1 per class
    3. stage2_cv_summary.png         -- bar chart of stress-head CV metrics
    4. stage2_cv_summary.txt         -- same numbers as text

Re-derives everything from the saved checkpoints and metadata directly
(same split logic/seed as training), so these numbers are always
consistent with whatever's actually in checkpoints/, not hand-copied.

Usage:
    python generate_evaluation_report.py
Output lands in: reports/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import UNIFIED_EMOTIONS
from audio_model.audio_model import AudioModel
from train_stage1 import load_stage1_metadata, speaker_disjoint_split, EmotionAudioDataset, collate_fn
from train_stage2_stress_v2 import (
    load_daic_metadata, precompute_multicrop_embeddings, train_head_on_fold, N_FOLDS,
)
from torch.utils.data import DataLoader

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1: emotion confusion matrix + classification report
# ---------------------------------------------------------------------------

def evaluate_stage1(device: str):
    print("=== Stage 1: Emotion Classification ===")
    df = load_stage1_metadata()
    _, val_df = speaker_disjoint_split(df)

    val_loader = DataLoader(
        EmotionAudioDataset(val_df), batch_size=32, shuffle=False, collate_fn=collate_fn,
    )

    model = AudioModel(num_emotions=len(UNIFIED_EMOTIONS)).to(device)
    state_dict = torch.load("checkpoints/stage1_best.pt", map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms = waveforms.to(device)
            output = model(waveforms)
            preds = output["emotion_logits"].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(UNIFIED_EMOTIONS))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(UNIFIED_EMOTIONS)))
    ax.set_yticks(range(len(UNIFIED_EMOTIONS)))
    ax.set_xticklabels(UNIFIED_EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(UNIFIED_EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stage 1 — Emotion Classification Confusion Matrix\n(CREMA-D + RAVDESS + SAVEE + TESS, val split)")
    for i in range(len(UNIFIED_EMOTIONS)):
        for j in range(len(UNIFIED_EMOTIONS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "stage1_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved {REPORTS_DIR / 'stage1_confusion_matrix.png'}")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=UNIFIED_EMOTIONS, digits=3)
    report_path = REPORTS_DIR / "stage1_classification_report.txt"
    report_path.write_text(
        "MindFlow Stage 1 — Emotion Classification Report\n"
        "Datasets: CREMA-D, RAVDESS, SAVEE, TESS (speaker-disjoint validation split)\n\n"
        + report
    )
    print(f"Saved {report_path}")
    print(report)


# ---------------------------------------------------------------------------
# Stage 2: stress head cross-validation summary
# ---------------------------------------------------------------------------

def evaluate_stage2(device: str):
    print("\n=== Stage 2: Stress Regression (5-Fold CV) ===")
    df = load_daic_metadata()

    model = AudioModel(num_emotions=len(UNIFIED_EMOTIONS)).to(device)
    state_dict = torch.load("checkpoints/stage1_best.pt", map_location=device, weights_only=True)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("stress_head")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    for p in model.parameters():
        p.requires_grad = False

    print("Precomputing embeddings for evaluation (same as training)...")
    embeddings, targets, binaries = precompute_multicrop_embeddings(model, df, device)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_maes, fold_corrs, fold_accs, fold_f1s = [], [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(embeddings, binaries.numpy()), start=1):
        mae, corr, acc, f1, _ = train_head_on_fold(
            embeddings, targets, binaries, train_idx, val_idx, device, epochs=100, lr=1e-3
        )
        fold_maes.append(mae)
        fold_corrs.append(corr)
        fold_accs.append(acc)
        fold_f1s.append(f1)

    metrics = {
        "MAE": (fold_maes, "lower is better"),
        "Pearson Corr": (fold_corrs, "higher is better"),
        "Binary Accuracy": (fold_accs, "higher is better"),
        "Binary F1": (fold_f1s, "higher is better"),
    }

    # Bar chart: mean +/- std per metric
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(metrics.keys())
    means = [np.mean(v[0]) for v in metrics.values()]
    stds = [np.std(v[0]) for v in metrics.values()]
    ax.bar(names, means, yerr=stds, capsize=6, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_title("Stage 2 — Stress Head, 5-Fold Cross-Validation (DAIC-WOZ)")
    ax.set_ylabel("Score")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "stage2_cv_summary.png", dpi=150)
    plt.close(fig)
    print(f"Saved {REPORTS_DIR / 'stage2_cv_summary.png'}")

    # Text summary
    lines = ["MindFlow Stage 2 — Stress Regression, 5-Fold Cross-Validation Summary",
             "Dataset: DAIC-WOZ (E-DAIC 2019), 266 sessions\n"]
    for name, (values, direction) in metrics.items():
        lines.append(f"{name}: {np.mean(values):.3f} +/- {np.std(values):.3f} ({direction})")
    lines.append("\nPer-fold breakdown:")
    for i in range(N_FOLDS):
        lines.append(
            f"  Fold {i+1}: MAE={fold_maes[i]:.3f} Corr={fold_corrs[i]:.3f} "
            f"Acc={fold_accs[i]:.3f} F1={fold_f1s[i]:.3f}"
        )
    summary_path = REPORTS_DIR / "stage2_cv_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"Saved {summary_path}")
    print("\n".join(lines))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_stage1(device)
    evaluate_stage2(device)
    print(f"\nAll evaluation artifacts saved to {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
