"""Siamese network: one shared CNN encoder + a dissimilarity function.

Two training modes (for the loss-comparison experiment):
  - "contrastive": distance between embeddings, trained with contrastive loss
  - "bce":         a small linear head on |e1 - e2|, trained with binary cross-entropy

In both modes `forward(a, b)` returns a DISSIMILARITY score (small = same character).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self, emb_dim: int = 64, mode: str = "contrastive"):
        super().__init__()
        assert mode in ("contrastive", "bce")
        self.mode = mode
        self.emb_dim = emb_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 105 -> 51
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 51 -> 24
            nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 24 -> 11
        )
        self.fc = nn.Linear(128 * 11 * 11, emb_dim)
        self.head = nn.Linear(emb_dim, 1) if mode == "bce" else None

    def embed(self, x):
        """Image batch (B,1,105,105) -> unit-length embeddings (B, emb_dim)."""
        z = self.fc(self.features(x).flatten(1))
        return F.normalize(z, dim=-1)

    def dissim(self, e1, e2):
        """Dissimilarity of embeddings; works with broadcasting (e.g. (1,1,D) vs (1,N,D))."""
        if self.mode == "contrastive":
            return (e1 - e2).norm(dim=-1)
        return -self.head((e1 - e2).abs()).squeeze(-1)  # -logit(same)

    def forward(self, a, b):
        return self.dissim(self.embed(a), self.embed(b))


def pair_loss(d, y, mode: str, margin: float = 1.0):
    """y = 1 for same character, 0 for different."""
    if mode == "contrastive":
        return (y * d.pow(2) + (1 - y) * F.relu(margin - d).pow(2)).mean()
    return F.binary_cross_entropy_with_logits(-d, y)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = SiameseNet(ckpt.get("emb_dim", 64), ckpt["mode"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
