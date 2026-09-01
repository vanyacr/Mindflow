"""
eval_temporal_vs_static.py — Does the BiGRU actually add value over the
frozen static model classifying frames independently and averaging?

Runs BOTH pipelines on the exact same val clips so the comparison is fair:
  (A) Temporal: frames -> frozen backbone embeddings -> BiGRU -> head
  (B) Static baseline: frames -> frozen backbone -> ORIGINAL static head
      -> softmax per frame -> average over the clip's frames
      (no GRU at all — this is "what if we just averaged frame predictions")

Also breaks down temporal accuracy by source (DFEW vs FERV39K) since they
have very different clip difficulty, which the combined number hides.

Run:
    python eval_temporal_vs_static.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path
import config_temporal as cfg
from datasets_temporal import build_temporal_dataset, get_transforms
from model_temporal import TemporalEmotionModel


def run_eval(ckpt_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    if ckpt_path is None:
        if cfg.CKPT_PATH_FROZEN.exists():
            ckpt_path = cfg.CKPT_PATH_FROZEN
        elif Path("checkpoints/best_model_temporal.pt").exists():
            ckpt_path = Path("checkpoints/best_model_temporal.pt")
        else:
            ckpt_path = cfg.CKPT_PATH_FROZEN

    ckpt_path = Path(ckpt_path)
    model = TemporalEmotionModel().to(device)
    model.static_model.to(device)

    state = torch.load(str(ckpt_path), map_location=device)
    model.gru.load_state_dict(state["gru"])
    model.attn.load_state_dict(state["attn"])
    model.head.load_state_dict(state["head"])
    if "residual_scale" in state:
        model.residual_scale.data.copy_(state["residual_scale"])
    if "backbone_last_block" in state:
        model.static_model.backbone.blocks[-1].load_state_dict(state["backbone_last_block"])
        print("  Loaded domain-adapted backbone last block")
    model.eval()
    print(f"Loaded temporal checkpoint: {ckpt_path}\n")

    print("Loading val sequences...")
    val_ds = build_temporal_dataset("val")
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=(device.type == "cuda"))

    # track which source each sample in the flattened val set came from,
    # so we can report DFEW vs FERV39K separately
    source_per_sample = []
    for ds in val_ds.datasets:
        source_per_sample.extend([ds.dataset_name] * len(ds))

    n_classes = cfg.NUM_CLASSES
    conf_temporal = np.zeros((n_classes, n_classes), dtype=np.int64)
    conf_static   = np.zeros((n_classes, n_classes), dtype=np.int64)

    # per-source correctness for BOTH pipelines
    source_correct_temporal = defaultdict(int)
    source_correct_static   = defaultdict(int)
    source_total            = defaultdict(int)

    sample_idx = 0
    print("\nRunning both pipelines on identical val clips...\n")

    with torch.no_grad():
        for seqs, soft_labels in tqdm(val_loader, desc="  eval", ncols=80):
            B, T, C, H, W = seqs.shape
            seqs = seqs.to(device)
            trues = soft_labels.argmax(dim=1).numpy()

            # ── (A) temporal pipeline — use the model's real forward pass
            # (attention pooling), not a hand-rolled mean-pool that would
            # silently diverge from what was actually trained ──
            temporal_logits = model(seqs)
            temporal_preds = temporal_logits.argmax(dim=1).cpu().numpy()

            # ── (B) static baseline: per-frame classify + average softmax ──
            flat = seqs.view(B * T, C, H, W)
            static_logits = model.static_model(flat)         # (B*T, 7)
            static_probs = F.softmax(static_logits, dim=1).view(B, T, n_classes)
            avg_probs = static_probs.mean(dim=1)              # (B, 7)
            static_preds = avg_probs.argmax(dim=1).cpu().numpy()

            for i in range(B):
                t, pt, ps = trues[i], temporal_preds[i], static_preds[i]
                conf_temporal[t, pt] += 1
                conf_static[t, ps]   += 1

                src = source_per_sample[sample_idx]
                source_total[src] += 1
                if pt == t:
                    source_correct_temporal[src] += 1
                if ps == t:
                    source_correct_static[src] += 1
                sample_idx += 1

    # ── overall accuracy ──
    temporal_acc = conf_temporal.diagonal().sum() / conf_temporal.sum()
    static_acc   = conf_static.diagonal().sum()   / conf_static.sum()

    print("\n" + "=" * 60)
    print(f"{'Pipeline':<30} {'Overall Acc':>12}")
    print("-" * 60)
    print(f"{'Temporal (BiGRU)':<30} {temporal_acc:>12.4f}")
    print(f"{'Static baseline (avg softmax)':<30} {static_acc:>12.4f}")
    print("=" * 60)

    lift = temporal_acc - static_acc
    if lift > 0.01:
        print(f"\nBiGRU adds +{lift:.4f} over static frame-averaging — real lift.")
    elif lift < -0.01:
        print(f"\nBiGRU is {abs(lift):.4f} WORSE than static frame-averaging — "
              "temporal modeling is not currently helping.")
    else:
        print(f"\nBiGRU and static frame-averaging are within noise of each other "
              f"({lift:+.4f}) — no clear lift from temporal modeling yet.")

    # ── per-class accuracy, both pipelines ──
    print(f"\n{'Class':<12} {'Temporal':>10} {'Static':>10} {'Delta':>8}")
    print("-" * 44)
    for i, emo in enumerate(cfg.EMOTIONS):
        t_acc = conf_temporal[i, i] / max(conf_temporal[i].sum(), 1)
        s_acc = conf_static[i, i]   / max(conf_static[i].sum(), 1)
        d = t_acc - s_acc
        sign = "+" if d >= 0 else ""
        print(f"{emo:<12} {t_acc:>10.3f} {s_acc:>10.3f} {sign}{d:>7.3f}")

    # ── per-source breakdown, BOTH pipelines ──
    print(f"\n{'Source':<12} {'Temporal Acc':>14} {'Static Acc':>12} {'N clips':>10}")
    print("-" * 52)
    for src in sorted(source_total):
        t_acc = source_correct_temporal[src] / max(source_total[src], 1)
        s_acc = source_correct_static[src]   / max(source_total[src], 1)
        print(f"{src:<12} {t_acc:>14.3f} {s_acc:>12.3f} {source_total[src]:>10}")

    print("\nIf the static baseline shows a similarly large DFEW-vs-FERV39K gap,")
    print("the bottleneck is the frozen backbone / domain shift, not temporal")
    print("modeling — pooling changes won't fix it. If static is roughly even")
    print("across sources but temporal isn't, the GRU itself is the problem.")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint file (default: best_model_temporal_frozen.pt)")
    parser.add_argument("--unfrozen", action="store_true",
                        help="Evaluate best_model_temporal_unfrozen.pt")
    args = parser.parse_args()

    ckpt = args.ckpt
    if args.unfrozen:
        ckpt = cfg.CKPT_PATH_UNFROZEN

    run_eval(ckpt_path=ckpt)
