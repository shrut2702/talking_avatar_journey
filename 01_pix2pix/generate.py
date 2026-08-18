"""Inference for pix2pix: load a trained generator, translate input images, plot.

Takes a single image or a folder of images (the input/mask side), not the paired
dataset. Reuses generate_imgs and show_input_output_grid from utils.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dataset import default_transform
from utils import generate_imgs, show_input_output_grid, load_generator

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_images(input_path, transform=None):
    """Load a single image file or every image in a folder.

    Returns a list of transformed tensors.
    """
    transform = transform or default_transform()
    path = Path(input_path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
    elif path.is_file():
        files = [path]
    else:
        files = []
    return [transform(Image.open(f).convert("RGB")) for f in files]


def generate(generator, samples_x, device=None, test_time_stats=True):
    """Translate a list of input tensors. Returns a list of CPU output tensors."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return [
        generate_imgs(generator, x.unsqueeze(0), device, test_time_stats).squeeze(0).cpu()
        for x in samples_x
    ]


def main():
    parser = argparse.ArgumentParser(description="Run a trained pix2pix generator on an image or folder.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint saved by train.py.")
    parser.add_argument("--input", required=True, help="Path to an input image or a folder of images.")
    parser.add_argument("--no-test-time-stats", action="store_true",
                        help="Use running BatchNorm stats (eval mode) instead of batch stats.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = load_images(args.input)
    if not inputs:
        raise SystemExit(f"No images found at: {args.input}")

    generator = load_generator(args.checkpoint, device)
    outputs = generate(generator, inputs, device, test_time_stats=not args.no_test_time_stats)

    show_input_output_grid(inputs, outputs)


if __name__ == "__main__":
    main()
