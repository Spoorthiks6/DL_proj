"""Shared helpers: paths, seeding, device, and image preprocessing for the demo."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

IMG_SIZE = 105  # Omniglot image size


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preprocess(img):
    """Turn a canvas drawing / uploaded image into a (1,1,105,105) tensor that looks
    like Omniglot: dark ink on a white background, character cropped and centred.

    Accepts a numpy array, a PIL image, or a Gradio editor dict. Returns None if blank.
    """
    if img is None:
        return None
    if isinstance(img, dict):  # Gradio Sketchpad / ImageEditor value
        img = img.get("composite")
        if img is None:
            return None
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> composite over white
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = arr[..., :3].astype(np.float32)
        arr = (rgb * alpha + 255.0 * (1.0 - alpha)).mean(axis=-1)
    elif arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=-1)
    else:
        arr = arr.astype(np.float32)

    if np.median(arr) < 127:  # dark background -> invert so ink is dark on white
        arr = 255.0 - arr

    ys, xs = np.where(arr < 200)  # ink pixels
    if len(ys) == 0:
        return None
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    pad = int(0.15 * side) + 1
    canvas = np.full((side + 2 * pad, side + 2 * pad), 255.0, dtype=np.float32)
    oy, ox = pad + (side - h) // 2, pad + (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    im = Image.fromarray(canvas.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
    return t.view(1, 1, IMG_SIZE, IMG_SIZE)
