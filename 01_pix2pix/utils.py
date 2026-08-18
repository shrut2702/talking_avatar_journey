"""Shared helpers: plotting, inference, and checkpoint save/load."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import Generator, PatchGAN


def plot_training_results(train_losses, val_losses, eval_interval, title, y_title):
    plt.figure(figsize=(6, 4))

    # Map to real global steps
    steps = (np.arange(len(train_losses))) * eval_interval

    plt.plot(steps, train_losses, label='Train', alpha=0.5)
    plt.plot(steps, val_losses, label='Val')

    plt.title(title)
    plt.xlabel('Global Steps')
    plt.ylabel(y_title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


def generate_imgs(generator, x, device, test_time_stats: bool = True):
    """Run the generator on a batch x.

    test_time_stats=True keeps the generator in train() mode so BatchNorm uses
    the current batch statistics instead of the running averages. See README.
    """
    if test_time_stats:
        generator.train()
    else:
        generator.eval()
    x = x.to(device)
    with torch.no_grad():
        return generator(x)


def show_translation_grid(inputs, targets, outputs):
    """Plot a 3-row grid: input masks, target photos, generated photos."""
    n = len(outputs)
    _, axes = plt.subplots(3, n, figsize=(12, 6))
    for i in range(n):
        axes[0][i].imshow(inputs[i].permute(1, 2, 0))
        axes[1][i].imshow(targets[i].permute(1, 2, 0))
        axes[2][i].imshow(outputs[i].permute(1, 2, 0))
    plt.show()


def show_input_output_grid(inputs, outputs):
    """Plot a 2-row grid: input images (top), generated outputs (bottom).

    Used at inference time when there is no ground-truth target to show.
    """
    n = len(outputs)
    _, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    axes = np.array(axes).reshape(2, n)  # keep indexing consistent when n == 1
    for i in range(n):
        axes[0][i].imshow(inputs[i].permute(1, 2, 0))
        axes[0][i].set_axis_off()
        axes[1][i].imshow(outputs[i].permute(1, 2, 0))
        axes[1][i].set_axis_off()
    plt.tight_layout()
    plt.show()


def save_checkpoint(generator, patchgan, path):
    torch.save(
        {"generator": generator.state_dict(), "patchgan": patchgan.state_dict()},
        path,
    )


def load_generator(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    return generator


def load_checkpoint(path, device=None):
    """Load both models. Returns (generator, patchgan)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    patchgan = PatchGAN().to(device)
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    patchgan.load_state_dict(ckpt["patchgan"])
    return generator, patchgan
