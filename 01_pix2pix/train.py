"""Training loop for pix2pix, extracted from the notebook into a train() function.

The notebook had three training cells that differed only in batch size, eval
frequency, and the test_time_stats flag. They are folded into this one function
via arguments. Calling train(train_dataset, test_dataset) with defaults
reproduces the first notebook run.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Generator, PatchGAN
from losses import (
    calc_discriminator_loss,
    calc_generator_loss,
    eval_discriminator_loss,
    eval_generator_loss,
)
from dataset import FacadeDataset, default_transform
from utils import generate_imgs, show_translation_grid, save_checkpoint


def train(
    train_dataset,
    test_dataset,
    *,
    epochs=50,
    batch_size=1,
    lr=2e-4,
    betas=(0.5, 0.999),
    l1_weight=100,
    eval_freq=50,
    eval_batch=50,
    test_time_stats=True,
    device=None,
    viz=False,
    sample_x=None,
    sample_y=None,
    log=True,
):
    """Train a pix2pix generator + PatchGAN.

    Returns (generator, patchgan, history), where history is a dict of the
    per-eval loss lists plus the eval_freq used to space them on the x-axis.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    bceloss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    generator = Generator().to(device)
    patchgan = PatchGAN().to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    disc_optimizer = torch.optim.Adam(patchgan.parameters(), lr=lr, betas=betas)

    history = {
        k: []
        for k in (
            "train_disc", "test_disc",
            "train_gen", "test_gen",
            "train_gen_l1", "test_gen_l1",
            "train_gen_l2", "test_gen_l2",
        )
    }
    history["eval_freq"] = eval_freq

    if viz and sample_x is None:
        sample_x = [test_dataset[i][0] for i in range(5)]
        sample_y = [test_dataset[i][1] for i in range(5)]

    global_steps = -1
    for epoch in range(epochs):
        for x, y in train_loader:
            generator.train()
            patchgan.train()

            # training discriminator
            disc_optimizer.zero_grad()
            disc_loss = calc_discriminator_loss(patchgan, generator, x, y, bceloss, device)
            disc_loss.backward()
            disc_optimizer.step()

            # training generator
            gen_optimizer.zero_grad()
            gen_loss, _, _ = calc_generator_loss(patchgan, generator, x, y, bceloss, l1_loss, l1_weight, device)
            gen_loss.backward()
            gen_optimizer.step()

            global_steps += 1

            if global_steps % eval_freq == 0:
                train_disc, test_disc = eval_discriminator_loss(
                    train_loader, test_loader, patchgan, generator, bceloss,
                    device, eval_batch, test_time_stats=test_time_stats,
                )
                (
                    train_gen, test_gen,
                    train_gen_l1, train_gen_l2,
                    test_gen_l1, test_gen_l2,
                ) = eval_generator_loss(
                    train_loader, test_loader, patchgan, generator, bceloss,
                    l1_loss, l1_weight, device, eval_batch,
                    test_time_stats=test_time_stats,
                )

                history["train_disc"].append(train_disc)
                history["test_disc"].append(test_disc)
                history["train_gen"].append(train_gen)
                history["test_gen"].append(test_gen)
                history["train_gen_l1"].append(train_gen_l1)
                history["test_gen_l1"].append(test_gen_l1)
                history["train_gen_l2"].append(train_gen_l2)
                history["test_gen_l2"].append(test_gen_l2)

                if log:
                    total_steps = epochs * len(train_loader)
                    print(f"Epoch: {epoch + 1:03d}/{epochs:03d}, Step: {global_steps + 1:06d}/{total_steps:06d}")
                    print(f"\t\tTrain Disc Loss: {train_disc:.4f}, Test Disc Loss: {test_disc:.4f}")
                    print(f"\t\tTrain Gen Loss: {train_gen:.4f}, Test Gen Loss: {test_gen:.4f}")

                if viz:
                    generated = [
                        generate_imgs(generator, s.unsqueeze(0), device, test_time_stats).squeeze(0).cpu()
                        for s in sample_x
                    ]
                    show_translation_grid(sample_x, sample_y, generated)

    return generator, patchgan, history


def main():
    parser = argparse.ArgumentParser(description="Train pix2pix on the CMP facade dataset.")
    parser.add_argument("--data", required=True, help="Path to the facade 'base' folder (paired .jpg/.png).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--l1-weight", type=float, default=100)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-batch", type=int, default=50)
    parser.add_argument("--no-test-time-stats", action="store_true",
                        help="Use running BatchNorm stats (eval mode) at inference instead of batch stats.")
    parser.add_argument("--out", default="pix2pix.pt", help="Checkpoint output path.")
    args = parser.parse_args()

    transform = default_transform()
    train_dataset = FacadeDataset(args.data, transforms=transform, train=True)
    test_dataset = FacadeDataset(args.data, transforms=transform, train=False)

    generator, patchgan, _ = train(
        train_dataset,
        test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_weight=args.l1_weight,
        eval_freq=args.eval_freq,
        eval_batch=args.eval_batch,
        test_time_stats=not args.no_test_time_stats,
    )

    save_checkpoint(generator, patchgan, args.out)
    print(f"Saved checkpoint to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
