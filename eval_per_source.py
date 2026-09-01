"""
eval_per_source.py — evaluates the trained checkpoint separately on each of
the 8 dataset sources' val split, instead of one blended number.

This tells us whether the 54.4% overall val accuracy is:
  (a) dfew/ferv39k being genuinely much harder (in-the-wild video frames vs
      curated static images) while the static sources are still ~74% — model
      is fine, the blended metric is just misleading, OR
  (b) accuracy dropped across ALL sources including the static ones — which
      would point to something actually broken (bad label mapping, LR/batch
      size mismatch, sampler issue) rather than just "harder data".

Run:
    python eval_per_source.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import config
from model import load_model
from datasets import (
    AffectNetDataset, CKPlusDataset, FERCk48Dataset, FERKaggle7Dataset,
    FERStockDataset, RAFDBDataset, DFEWFramesDataset, FERV39KFramesDataset,
    get_transforms,
)

SOURCES = {
    "affectnet": AffectNetDataset,
    "ckplus":    CKPlusDataset,
    "fer_ck48":  FERCk48Dataset,
    "fer_k7":    FERKaggle7Dataset,
    "fer_stock": FERStockDataset,
    "rafdb":     RAFDBDataset,
    "dfew":      DFEWFramesDataset,
    "ferv39k":   FERV39KFramesDataset,
}


def evaluate_one(model, dataset, device, batch_size=96):
    if len(dataset) == 0:
        return None, 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=(device.type == "cuda"))
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, soft_labels in tqdm(loader, desc="  eval", leave=False, ncols=80):
            imgs = imgs.to(device)
            soft_labels = soft_labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            trues = soft_labels.argmax(dim=1)
            correct += (preds == trues).sum().item()
            total += imgs.size(0)
    return correct / max(total, 1), total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, device = load_model(config.CKPT_PATH, device)

    tf = get_transforms("val")

    print("\n" + "=" * 60)
    print(f"{'Source':<12} {'Val Acc':>9} {'Samples':>9}")
    print("-" * 60)

    results = {}
    for name, cls in SOURCES.items():
        ds = cls("val", tf)
        acc, n = evaluate_one(model, ds, device)
        results[name] = (acc, n)
        if acc is None:
            print(f"{name:<12} {'—':>9} {0:>9}  (0 samples — skipped)")
        else:
            print(f"{name:<12} {acc*100:>8.2f}% {n:>9}")

    print("=" * 60)

    static = ["affectnet", "ckplus", "fer_ck48", "fer_k7", "fer_stock", "rafdb"]
    video  = ["dfew", "ferv39k"]

    static_correct = sum(results[s][0] * results[s][1] for s in static if results[s][0] is not None)
    static_total   = sum(results[s][1] for s in static if results[s][0] is not None)
    video_correct  = sum(results[s][0] * results[s][1] for s in video if results[s][0] is not None)
    video_total    = sum(results[s][1] for s in video if results[s][0] is not None)

    print(f"\nStatic sources combined : {100*static_correct/max(static_total,1):.2f}%  ({static_total} samples)")
    print(f"Video sources combined  : {100*video_correct/max(video_total,1):.2f}%  ({video_total} samples)")
    print("\nIf static is still ~70%+ and video is much lower, the model is fine —")
    print("the blended metric was just misleading. If static also dropped hard,")
    print("something's actually broken and we look at label mapping / LR next.")


if __name__ == "__main__":
    main()
