"""'Teach it once' demo.   Run:  python app.py --mode contrastive

Tabs:
  1. Teach      - draw a symbol once, give it a name
  2. Recognize  - draw it again; top-3 matches, or "Unknown" if too far from everything taught
  3. Compare    - are two images the same character? (draw)
"""
import argparse
import json

import gradio as gr
import torch

from src.model import load_model
from src.utils import CKPT_DIR, RESULTS_DIR, ROOT, get_device, preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="contrastive", choices=["contrastive", "bce"])
parser.add_argument("--device", default="auto")
parser.add_argument("--share", action="store_true")
args = parser.parse_args()

device = get_device(args.device)
ckpt = CKPT_DIR / f"siamese_{args.mode}.pt"
if not ckpt.exists():
    raise SystemExit(f"{ckpt} not found. Train first:  python -m src.train --mode {args.mode}")
model = load_model(ckpt, device)

thr_file = RESULTS_DIR / f"threshold_{args.mode}.json"
DEFAULT_THR = json.loads(thr_file.read_text())["threshold"] if thr_file.exists() else 0.8
SUPPORT_FILE = ROOT / "support_set.pt"
support = torch.load(SUPPORT_FILE) if SUPPORT_FILE.exists() else {}  # name -> (k, D) embeddings
@torch.no_grad()
def embed_img(img):
    t = preprocess(img)
    return None if t is None else model.embed(t.to(device))  # (1, D)


def table():
    return [[n, len(v)] for n, v in support.items()]


def teach(img, name):
    name = (name or "").strip()
    if not name:
        return "Give the symbol a name first.", table()
    e = embed_img(img)
    if e is None:
        return "The canvas is empty - draw a symbol first.", table()
    support[name] = torch.cat([support[name], e.cpu()]) if name in support else e.cpu()
    torch.save(support, SUPPORT_FILE)
    return f"Learned '{name}' ({len(support[name])} example(s)). Now try the Recognize tab.", table()


def clear_support():
    support.clear()
    if SUPPORT_FILE.exists():
        SUPPORT_FILE.unlink()
    return "Cleared all taught symbols.", table()


def recognize(img, thr):
    if not support:
        return "Teach at least one symbol first.", []
    e = embed_img(img)
    if e is None:
        return "The canvas is empty - draw a symbol first.", []
    e = e.cpu()
    scores = []
    for name, embs in support.items():  # closest example per taught symbol
        d = model.dissim(e.to(device)[:, None, :], embs.to(device)[None, :, :]).min().item()
        scores.append((name, d))
    scores.sort(key=lambda s: s[1])
    top = [[n, round(d, 3)] for n, d in scores[:3]]
    if scores[0][1] > thr:
        verdict = f"**Unknown symbol.** Nothing taught is close enough (best distance {scores[0][1]:.3f} > threshold {thr:.2f})."
    else:
        verdict = f"**It's '{scores[0][0]}'** (distance {scores[0][1]:.3f} <= threshold {thr:.2f})."
    return verdict, top


def compare(img1, img2, thr):
    e1, e2 = embed_img(img1), embed_img(img2)
    if e1 is None or e2 is None:
        return "Provide two non-empty images."
    d = model.dissim(e1, e2).item()
    same = d <= thr
    return f"**{'SAME' if same else 'DIFFERENT'} character** (distance {d:.3f}, threshold {thr:.2f})"


brush = gr.Brush(default_size=3, colors=["#000000"], color_mode="fixed")


def canvas(label):
    return gr.Sketchpad(label=label, type="numpy", brush=brush, canvas_size=(300, 300))


with gr.Blocks(title="Learn Once, Recognize Anywhere") as demo:
    gr.Markdown("# Learn Once, Recognize Anywhere\nOne-shot handwriting matching with a Siamese network. "
                "Tip: draw with thin strokes, one symbol per canvas.")
    lo, hi = (0.0, 2.0) if args.mode == "contrastive" else (-10.0, 10.0)
    thr = gr.Slider(lo, hi, value=DEFAULT_THR, step=0.01,
                    label="Match threshold (distance <= threshold means 'same'; auto-picked on validation data)")
    with gr.Tab("1. Teach"):
        c1 = canvas("Draw a new symbol")
        name = gr.Textbox(label="Symbol name")
        with gr.Row():
            b1 = gr.Button("Teach it", variant="primary")
            b1c = gr.Button("Forget everything")
        msg1 = gr.Markdown()
        tbl = gr.Dataframe(headers=["Symbol", "# examples"], value=table(), interactive=False)
        b1.click(teach, [c1, name], [msg1, tbl])
        b1c.click(clear_support, None, [msg1, tbl])
    with gr.Tab("2. Recognize"):
        c2 = canvas("Draw it again")
        b2 = gr.Button("Recognize", variant="primary")
        verdict = gr.Markdown()
        top3 = gr.Dataframe(headers=["Closest symbol", "Distance"], interactive=False)
        b2.click(recognize, [c2, thr], [verdict, top3])
    with gr.Tab("3. Compare two"):
        with gr.Row():
            i1 = canvas("Image A (draw)")
            i2 = canvas("Image B (draw)")
        b3 = gr.Button("Same or different?", variant="primary")
        out3 = gr.Markdown()
        b3.click(compare, [i1, i2, thr], out3)
if __name__ == "__main__":
    demo.launch(share=args.share)
