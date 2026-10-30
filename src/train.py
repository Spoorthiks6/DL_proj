"""Train the Siamese network.

Usage:
    python -m src.train --mode contrastive --epochs 15
    python -m src.train --mode bce --epochs 15      # for the loss comparison
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .dataset import EVAL_TF, TRAIN_TF, OmniglotSet, PairDataset, split_characters
from .evaluate import embed_all, n_way_accuracy
from .model import SiameseNet, pair_loss
from .utils import CKPT_DIR, RESULTS_DIR, get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="contrastive", choices=["contrastive", "bce"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--pairs_per_epoch", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--val_chars", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"device={device}  mode={args.mode}")

    train_base = OmniglotSet(True, TRAIN_TF)   # augmented images for training
    eval_base = OmniglotSet(True, EVAL_TF)     # clean images for validation
    train_classes, val_classes = split_characters(train_base, args.val_chars, args.seed)
    print(f"train characters: {len(train_classes)}   validation characters: {len(val_classes)}")
    val_idx = [i for c in val_classes for i in eval_base.by_class[c]]

    loader = DataLoader(PairDataset(train_base, train_classes, args.pairs_per_epoch),
                        batch_size=args.batch_size, num_workers=args.workers, drop_last=True)
    model = SiameseNet(64, args.mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history, best = [], -1.0
    ckpt_path = CKPT_DIR / f"siamese_{args.mode}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for a, b, y in loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            loss = pair_loss(model(a, b), y, args.mode, args.margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)

        model.eval()
        v_embs, v_labs = embed_all(model, eval_base, val_idx, device)
        v5 = n_way_accuracy(model, v_embs, v_labs, 5, 300, seed=0)
        v20 = n_way_accuracy(model, v_embs, v_labs, 20, 300, seed=0)
        history.append(dict(epoch=epoch, loss=total / n, val_5way=v5, val_20way=v20))
        flag = ""
        if v20 > best:
            best = v20
            torch.save({"state_dict": model.state_dict(), "mode": args.mode, "emb_dim": 64}, ckpt_path)
            flag = "  <- saved"
        print(f"epoch {epoch:02d}  loss {total/n:.4f}  val 5-way {v5:.3f}  val 20-way {v20:.3f}  "
              f"({time.time()-t0:.0f}s){flag}")

    (RESULTS_DIR / f"history_{args.mode}.json").write_text(json.dumps(history, indent=2))
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["loss"] for h in history], marker="o")
    ax[0].set(title=f"Training loss ({args.mode})", xlabel="epoch", ylabel="loss")
    ax[1].plot(ep, [h["val_5way"] for h in history], marker="o", label="5-way")
    ax[1].plot(ep, [h["val_20way"] for h in history], marker="o", label="20-way")
    ax[1].set(title="Validation one-shot accuracy", xlabel="epoch", ylabel="accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"training_curve_{args.mode}.png", dpi=150)
    print(f"best val 20-way: {best:.3f}  ->  {ckpt_path}")


if __name__ == "__main__":
    main()
