# Learn Once, Recognize Anywhere
**One-shot handwritten character recognition with a Siamese network (Omniglot)**
Deep Learning mini project - *Verification Network using Siamese Architecture (Few-Shot Prediction)*

## What it does
A Siamese network learns to answer one question: **"are these two handwritten characters the same?"**
Because it learns *similarity* instead of fixed classes, it can recognize characters it never saw
in training after seeing a **single** example.

### Unique features
| Feature | Where |
|---|---|
| **"Teach it once" drawing canvas** - draw a brand-new symbol once, then recognize it | `app.py` tabs 1-2 |
| **"Unknown" rejection** - refuses to guess if nothing taught is close enough; threshold auto-picked on validation data and adjustable with a slider | `app.py`, `src/evaluate.py` |
| **Top-3 matches with distances** - explains the decision | `app.py` tab 2 |
| **Loss comparison** - contrastive vs binary cross-entropy | `--mode contrastive / bce` |
| **Verification metrics** - FAR / FRR, not just accuracy | `src/evaluate.py` |

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Omniglot downloads automatically (~10 MB) on the first run.

## Run (in this order)
```bash
# 1. Train (saves best checkpoint to checkpoints/, curves to results/)
python -m src.train --mode contrastive --epochs 15

# 2. Evaluate on unseen characters (writes results/metrics_*.json and the threshold)
python -m src.evaluate --mode contrastive

# 3. (Optional) loss comparison
python -m src.train --mode bce --epochs 15
python -m src.evaluate --mode bce
python -m src.evaluate --table

# 4. Launch the demo
python app.py --mode contrastive
```
Slow on CPU? Use `--pairs_per_epoch 8000 --epochs 10`. On Windows, if data loading errors, add `--workers 0`.

## Method (short version)
- **Encoder:** 3 x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool] (32, 64, 128 channels) -> Linear -> **64-d embedding**, L2-normalised. The *same* encoder (shared weights) processes both images.
- **Distance:** `d = || f(A) - f(B) ||`  (small = same character)
- **Contrastive loss:** `L = y * d^2 + (1 - y) * max(0, m - d)^2`, margin `m = 1`
  Same pairs (`y=1`) are pulled together; different pairs (`y=0`) are pushed apart until they are at least `m` away.
- **Optimizer:** Adam, lr 1e-3. Light augmentation (small rotation / shift / scale) helps the model handle canvas drawings.
- **Data split (by character, never by image):**
  - Train: 864 of the 964 `background` characters
  - Validation: 100 held-out `background` characters (checkpoint choice + threshold)
  - Test: the 659 `evaluation` characters from 20 unseen alphabets
- **One-shot N-way test:** give 1 example of each of N unseen characters, show a new image of one of them, pick the smallest distance. Reported for N = 5 and 20.
- **Unknown rejection:** if the nearest taught symbol is farther than the threshold, answer "Unknown".

## Project structure
```
siamese-omniglot/
├── app.py                 # Gradio demo (teach / recognize / compare)
├── requirements.txt
├── README.md
├── src/
│   ├── dataset.py         # Omniglot, character splits, pair generator
│   ├── model.py           # Siamese CNN, losses
│   ├── train.py           # training loop + curves
│   ├── evaluate.py        # N-way accuracy, threshold, FAR/FRR, table
│   └── utils.py           # paths, seeding, canvas preprocessing
├── data/                  # Omniglot (auto-downloaded)
├── checkpoints/           # siamese_<mode>.pt
└── results/               # curves, metrics_*.json
```

## Tips for the demo
- Draw with thin strokes, one symbol per canvas, filling most of the canvas.
- Teach 2-3 *very different* symbols first; similar-looking ones are harder.
- If everything says "Unknown", raise the threshold slider a little; if wrong matches are accepted, lower it.
- Hand-drawn mouse strokes differ from Omniglot's pen strokes, so expect some drop vs. the test-set numbers. That gap is a good "limitations" point for the report.

## Report outline
1. Problem and motivation (few-shot learning, verification) 2. Dataset and split
3. Architecture and loss (diagram + formula) 4. Experiments: training curve, 5-way / 20-way accuracy, FAR/FRR, contrastive vs BCE table
5. Visualization: t-SNE, top-3 examples, unknown rejection 6. Demo screenshots
7. Limitations and future work (triplet loss, pretrained backbone, more scripts)
