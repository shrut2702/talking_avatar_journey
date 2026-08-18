"""Loss functions and loss-over-loader evaluation helpers for pix2pix.

The generator loss is a weighted sum of an adversarial BCE term (fool the
discriminator) and an L1 term (stay close to the target pixel-wise). The
discriminator loss is the average BCE over real and fake pairs.
"""

import torch


def calc_discriminator_loss(patchgan, generator, x, y, bceloss, device):
    x = x.to(device)
    y = y.to(device)
    fake_img_pred = patchgan(torch.cat((x, generator(x).detach()), dim=1))
    fake_img_target = torch.zeros_like(fake_img_pred)
    l1 = bceloss(fake_img_pred, fake_img_target)  # avg per pixel and not per sample
    real_img_pred = patchgan(torch.cat((x, y), dim=1))
    real_img_target = torch.ones_like(real_img_pred)
    l2 = bceloss(real_img_pred, real_img_target)  # avg per pixel and not per sample
    return (l1 + l2) / 2


def calc_generator_loss(patchgan, generator, x, y, bceloss, l1_loss, l1_weight, device):
    x = x.to(device)
    y = y.to(device)
    generated_img = generator(x)
    disc_op = patchgan(torch.cat((x, generated_img), dim=1))
    target = torch.ones_like(disc_op)
    l1 = l1_weight * l1_loss(generated_img, y)
    l2 = bceloss(disc_op, target)
    return l1 + l2, l1, l2


def calc_loader_disc_loss(loader, patchgan, generator, bceloss, device, num_batch=50):
    if len(loader) == 0:
        return float('nan')
    elif num_batch is None:
        num_batch = len(loader)
    elif num_batch <= 0:
        return float('nan')
    else:
        num_batch = min(num_batch, len(loader))

    total_loss = 0

    for i, (x, y) in enumerate(loader):
        if i < num_batch:
            with torch.no_grad():
                disc_loss = calc_discriminator_loss(patchgan, generator, x, y, bceloss, device)
                total_loss += disc_loss.item()
        else:
            break

    return total_loss / num_batch


def calc_loader_gen_loss(loader, patchgan, generator, bceloss, l1_loss, l1_weight, device, num_batch=50):
    if len(loader) == 0:
        return float('nan')
    elif num_batch is None:
        num_batch = len(loader)
    elif num_batch <= 0:
        return float('nan')
    else:
        num_batch = min(num_batch, len(loader))

    total_loss = 0
    total_l1_loss = 0
    total_l2_loss = 0

    for i, (x, y) in enumerate(loader):
        if i < num_batch:
            with torch.no_grad():
                gen_loss, l1, l2 = calc_generator_loss(patchgan, generator, x, y, bceloss, l1_loss, l1_weight, device)
                total_loss += gen_loss.item()
                total_l1_loss += l1.item()
                total_l2_loss += l2.item()
        else:
            break

    return total_loss / num_batch, total_l1_loss / num_batch, total_l2_loss / num_batch


def eval_discriminator_loss(train_loader, test_loader, patchgan, generator, bceloss, device, num_batch=50, test_time_stats: bool = True):
    patchgan.eval()
    if test_time_stats:
        generator.train()
    else:
        generator.eval()
    train_loss = calc_loader_disc_loss(train_loader, patchgan, generator, bceloss, device, num_batch)
    test_loss = calc_loader_disc_loss(test_loader, patchgan, generator, bceloss, device, num_batch)

    return train_loss, test_loss


def eval_generator_loss(train_loader, test_loader, patchgan, generator, bceloss, l1_loss, l1_weight, device, num_batch=50, test_time_stats: bool = True):
    patchgan.eval()
    if test_time_stats:
        generator.train()
    else:
        generator.eval()
    train_loss, train_l1_loss, train_l2_loss = calc_loader_gen_loss(train_loader, patchgan, generator, bceloss, l1_loss, l1_weight, device, num_batch)
    test_loss, test_l1_loss, test_l2_loss = calc_loader_gen_loss(test_loader, patchgan, generator, bceloss, l1_loss, l1_weight, device, num_batch)

    return train_loss, test_loss, train_l1_loss, train_l2_loss, test_l1_loss, test_l2_loss
