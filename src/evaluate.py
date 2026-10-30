"""Evaluation: N-way one-shot accuracy, same/different threshold, FAR/FRR.

Usage:
    python -m src.evaluate --mode contrastive
    python -m src.evaluate --mode bce
    python -m src.evaluate --table          # print loss-comparison table
"""
import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import EVAL_TF, OmniglotSet, split_characters
from .model import load_model
from .utils import CKPT_DIR, RESULTS_DIR, get_device, set_seed


@torch.no_grad()
def embed_all(model, dset, indices=None, device="cpu", bs=256):
    """Embed images once; returns (embeddings on `device`, labels)."""
    model.eval()
    idx = list(range(len(dset))) if indices is None else list(indices)
    loader = DataLoader(Subset(dset, idx), batch_size=bs, shuffle=False, num_workers=0)
    embs, labs = [], []
    for x, y in loader:
        embs.append(model.embed(x.to(device)))
        labs.append(y)
    return torch.cat(embs), torch.cat(labs)


def _by_class(labels):
    by = {}
    for i, l in enumerate(labels.tolist()):
        by.setdefault(l, []).append(i)
    return by


@torch.no_grad()
def n_way_accuracy(model, embs, labels, n=20, trials=1000, seed=0):
    """One-shot N-way test: 1 support image for each of N characters, classify a query
    (a different image of one of them) by the smallest dissimilarity."""
    rng = random.Random(seed)
    by = _by_class(labels)
    classes = list(by)
    correct = 0
    for _ in range(trials):
        cs = rng.sample(classes, n)
        support = [rng.choice(by[c]) for c in cs]
        t = rng.randrange(n)
        query = rng.choice([i for i in by[cs[t]] if i != support[t]])
        d = model.dissim(embs[query][None, None, :], embs[support][None, :, :])  # (1, n)
        correct += int(d.argmin().item() == t)
    return correct / trials


@torch.no_grad()
def sample_pair_scores(model, embs, labels, n_pairs=10000, seed=0):
    """Dissimilarities for random same / different pairs (50/50). Returns (d, y) numpy."""
    rng = random.Random(seed)
    by = _by_class(labels)
    classes = list(by)
    i1s, i2s, ys = [], [], []
    for _ in range(n_pairs):
        c1 = rng.choice(classes)
        if rng.random() < 0.5:
            a, b = rng.sample(by[c1], 2)
            y = 1
        else:
            c2 = rng.choice(classes)
            while c2 == c1:
                c2 = rng.choice(classes)
            a, b, y = rng.choice(by[c1]), rng.choice(by[c2]), 0
        i1s.append(a); i2s.append(b); ys.append(y)
    d = model.dissim(embs[i1s], embs[i2s]).cpu().numpy()
    return d, np.array(ys)


def best_threshold(d, y):
    """Threshold t maximising accuracy of: predict 'same' if d <= t."""
    cands = np.quantile(d, np.linspace(0, 1, 400))
    accs = [((d <= t) == (y == 1)).mean() for t in cands]
    k = int(np.argmax(accs))
    return float(cands[k]), float(accs[k])


def verification_metrics(d, y, t):
    pred_same = d <= t
    acc = float((pred_same == (y == 1)).mean())
    far = float(pred_same[y == 0].mean())      # different pair wrongly accepted
    frr = float((~pred_same)[y == 1].mean())   # same pair wrongly rejected
    return acc, far, frr


def print_table():
    rows = []
    for mode in ("contrastive", "bce"):
        p = RESULTS_DIR / f"metrics_{mode}.json"
        if p.exists():
            rows.append((mode, json.loads(p.read_text())))
    if not rows:
        print("No metrics yet - run evaluate for each mode first.")
        return
    print(f"{'loss':<14}{'5-way':>8}{'20-way':>8}{'pair acc':>10}{'FAR':>8}{'FRR':>8}")
    for mode, m in rows:
        print(f"{mode:<14}{m['test_5way']*100:>7.1f}%{m['test_20way']*100:>7.1f}%"
              f"{m['pair_acc']*100:>9.1f}%{m['far']*100:>7.1f}%{m['frr']*100:>7.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="contrastive", choices=["contrastive", "bce"])
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val_chars", type=int, default=100)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if args.table:
        return print_table()

    set_seed(42)
    device = get_device(args.device)
    ckpt_path = CKPT_DIR / f"siamese_{args.mode}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found - run: python -m src.train --mode {args.mode}")
    model = load_model(ckpt_path, device)

    # 1) pick the same/different threshold on VALIDATION characters (never the test set)
    bg = OmniglotSet(True, EVAL_TF)
    _, val_classes = split_characters(bg, args.val_chars)
    val_idx = [i for c in val_classes for i in bg.by_class[c]]
    v_embs, v_labs = embed_all(model, bg, val_idx, device)
    vd, vy = sample_pair_scores(model, v_embs, v_labs, seed=1)
    thr, val_acc = best_threshold(vd, vy)

    # 2) report on the TEST set: 659 characters from 20 alphabets never seen in training
    test = OmniglotSet(False, EVAL_TF)
    t_embs, t_labs = embed_all(model, test, None, device)
    a5 = n_way_accuracy(model, t_embs, t_labs, 5, args.trials, seed=2)
    a20 = n_way_accuracy(model, t_embs, t_labs, 20, args.trials, seed=3)
    td, ty = sample_pair_scores(model, t_embs, t_labs, seed=4)
    pair_acc, far, frr = verification_metrics(td, ty, thr)

    out = dict(mode=args.mode, threshold=thr, val_pair_acc=val_acc, test_5way=a5, test_20way=a20,
               pair_acc=pair_acc, far=far, frr=frr, trials=args.trials)
    (RESULTS_DIR / f"metrics_{args.mode}.json").write_text(json.dumps(out, indent=2))
    # threshold file used by the demo app for "unknown" rejection
    (RESULTS_DIR / f"threshold_{args.mode}.json").write_text(json.dumps({"threshold": thr}))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
