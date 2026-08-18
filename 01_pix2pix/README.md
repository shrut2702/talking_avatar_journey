# Pix2Pix (from scratch)

A from-scratch PyTorch implementation of pix2pix, trained to turn architectural
facade masks into photo-like facades. Pix2pix is a general recipe for
image-to-image translation, so the same setup works for maps to satellite,
edges to photo, day to night, and so on. Only the paired dataset changes.

## Intuition

The model is a conditional GAN. A generator takes the input image and produces a
translated one. A discriminator looks at the pair and judges whether the second
image is a real match for the first or something the generator faked. The two
train against each other.

**Why GAN and not only L1/L2 loss?**

L1 and L2 losses models and captures underlying structure of the images well but is bad at capturing sharper details. Inherently, these losses have mean seeking behaviour, and therefore results in blurry images. GAN, on the other hand, has mode seeking behaviour i.e. it would rather generate a few real looking images than all blurry images.

**Why does the PatchGAN take the input pair (mask + ground truth/generated image) and not only the image to be tested?**

We want the generator to produce real and grounded to input images. Without conditioning discriminator on input images, the generator will produce real looking image that is entirely different from the input mask and still goes without getting penalized.

**Why two losses for generator?**

The loss has two jobs, and they split along frequency.

**L1 loss handles the low frequencies.** Comparing the output to the target
pixel by pixel pushes the generator to get the broad structure right: where the
walls, windows, and sky are, and roughly what color. L1 on its own gets you
there, but the result looks blurry, because when the model is unsure it hedges
by averaging, and averaging smears fine detail.

**The PatchGAN handles the high frequencies.** This is the discriminator, and
the trick is that it does not output one real/fake number for the whole image.
It outputs a grid of scores, one per patch. Each score judges a small local
region. That focuses it on local texture such as edges, bricks, and window
frames, which is exactly the sharp detail L1 washes out. The generator has to
make every patch look locally real, not just globally close.

So the division of labor is: L1 keeps the output faithful to the target's
overall layout, and the PatchGAN keeps it sharp and textured. Together they give
you something both correct and crisp.

## Why does pix2pix use test-time BatchNorm stats?

At inference the generator stays in `train()` mode for BatchNorm, so each image
is normalized by its own statistics instead of the running mean and variance
saved during training. The authors do this on purpose. Here is the reasoning
worked out from the implementation, plus a small experiment that backs it up.

**The per-image angle.** Normalizing each image by its own stats is effectively
instance normalization. It strips out image-specific contrast and color, which
is what you want for a translation task: the output should follow the input's
structure, not inherit whatever color statistics the training set averaged out
to. That is the "use per-image stats to strip color and artifacts" reason.

**Why batch size 1 forces it.** This model trains at `batch_size=1`, so
BatchNorm normalizes each image on its own. During training the network only
ever sees perfectly zero-mean, unit-variance activations, because every single
example is normalized to exactly that. It never learns to cope with anything
else. Switch to aggregated (running) stats at inference and each image is now
normalized by the global mean and variance, not its own, so its activations land
at some offset from the perfect 0/1 the network is used to. It has never seen
offset activations, so it handles them badly and you get color shifts and
artifacts. Using the image's own stats at test time keeps activations at 0/1,
matching training, so the output stays clean.

**Hypothesis: bigger batches should soften this.** With batch size 8 or 16,
BatchNorm normalizes each image by the shared batch stats rather than its own.
Individual images no longer sit at a perfect 0/1; they scatter at small offsets
around the batch statistics. If the network trains on activations that are
already slightly off, it should learn to tolerate offsets, and inference on
global stats should hold up better.

**Experiment.** Retrained the same model at batch size 8 and compared inference
using aggregated running stats against per-image test-time stats.

**Result.** The larger batch did mitigate the problem, to a point. Inference with
aggregated training stats looked better than it did at batch size 1, but still
did not fully match the test-time-stats output.

**Conclusion.** The `batch_size=1` model could not handle activations that sat
even slightly off the normalized distribution, because it had never seen any.
Bigger training batches expose it to that offset and buy back some robustness,
which is why the gap shrinks but does not fully close.

## Files

- `models.py` - `Generator` (U-Net) and `PatchGAN` discriminator.
- `dataset.py` - `FacadeDataset` (pairs each `.png` mask with its `.jpg` photo) and `default_transform`.
- `losses.py` - generator and discriminator losses, plus helpers that average a loss over a loader.
- `utils.py` - plotting, `generate_imgs`, a 3-row result grid, and checkpoint save/load.
- `train.py` - the `train(...)` function and a command-line entry point.
- `generate.py` - load a checkpoint, translate samples, plot inputs / targets / outputs.
- `pix2pix.ipynb` - the original notebook this was extracted from.

## Train

From the command line:

```bash
python train.py --data /path/to/CMP_facade_DB_base/base --epochs 50 --out pix2pix.pt
```

Or from Python:

```python
from dataset import FacadeDataset, default_transform
from train import train

t = default_transform()
train_ds = FacadeDataset(".../base", transforms=t, train=True)
test_ds  = FacadeDataset(".../base", transforms=t, train=False)

generator, patchgan, history = train(train_ds, test_ds, epochs=50, viz=True)
```

`train(...)` returns the two models and a `history` dict of loss curves. Pass
them to `plot_training_results` in `utils.py` to see the curves.

## Generate

Point it at a single input image or a folder of them (no dataset needed):

```bash
python generate.py --checkpoint pix2pix.pt --input /path/to/mask.png
python generate.py --checkpoint pix2pix.pt --input /path/to/masks_folder/
```

Or in Python:

```python
from utils import load_generator
from generate import load_images, generate

g = load_generator("pix2pix.pt")
inputs = load_images("/path/to/masks_folder/")   # or a single image path
outputs = generate(g, inputs)                     # list of translated images
```

It plots a 2-row grid: inputs on top, generated outputs below.

## Notes

A few choices here differ from the original paper. They are kept as written in
the notebook, not silently changed.

- **Output range is [0, 1], not [-1, 1].** The generator ends in a Sigmoid and
  the data is only scaled by `ToTensor`. The paper uses a Tanh output with data
  normalized to [-1, 1]. Both train; this one just matches the notebook.
- **Sigmoid + `BCELoss`, not `BCEWithLogitsLoss`.** The discriminator applies a
  Sigmoid and the loss is plain BCE. Fusing them into `BCEWithLogitsLoss` would
  be more numerically stable, but the behavior here is the notebook's.
- **`test_time_stats` and BatchNorm.** When `test_time_stats=True`, the
  generator stays in `train()` mode during inference, so BatchNorm uses the
  current batch's statistics instead of its running averages. The notebook ran
  the training three times to compare this against eval-mode inference and
  against a larger batch size. Those three runs are folded into `train(...)`
  through the `batch_size`, `eval_freq`, and `test_time_stats` arguments.
- **Dataset returns `(mask, img)`,** so the mask is the input `x` and the photo
  is the target `y`.


