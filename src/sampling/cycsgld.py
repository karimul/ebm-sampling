import torch
from autograd.numpy import sqrt, cos, pi
import numpy as np

def gen_image_cycsgld(label, FLAGS, model, im_neg, num_steps, sample=False):
    im_noise = torch.randn_like(im_neg).detach()
    total=1e6
    cycles=3500
    sub_total = total / cycles
    T = 1e-7
    # noise_scale = 0.25
    # total=1e6
    # cycles=5000
    # sub_total = total / cycles
    # T = 1e-6
    
    im_negs_samples = []

    for i in range(num_steps):
        im_noise.normal_()
        iters = i
        r_remainder = (iters % sub_total) * 1.0 / sub_total
        cyc_lr = FLAGS.step_lr * 5 / 2 * (cos(pi * r_remainder) + 1)
        # print("\ncyc_lr", cyc_lr)

        if FLAGS.anneal:
            im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * im_noise
        else:
            # im_neg = im_neg + 0.001 * im_noise
            im_neg = im_neg + sqrt(2 * cyc_lr * T) * FLAGS.noise_scale * im_noise
        # print("\nnoise_cyc_lr", sqrt(2 * cyc_lr * T) * noise_scale)
        im_neg.requires_grad_(requires_grad=True)
        energy = model.forward(im_neg, label)

        if FLAGS.all_step:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        else:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg])[0]

        if i == num_steps - 1:
            im_neg_orig = im_neg
            im_neg = im_neg - cyc_lr * im_grad

            if FLAGS.dataset in ("cifar10", "celeba", "cats"):
                n = 128
            elif FLAGS.dataset == "celebahq":
                # Save space
                n = 128
            elif FLAGS.dataset == "lsun":
                # Save space
                n = 32
            elif FLAGS.dataset == "object":
                # Save space
                n = 32
            elif FLAGS.dataset == "mnist":
                n = 128
            elif FLAGS.dataset == "imagenet":
                n = 32
            elif FLAGS.dataset == "stl":
                n = 32

            im_neg_kl = im_neg_orig[:n]
            if sample:
                pass
            else:
                energy = model.forward(im_neg_kl, label)
                im_grad = torch.autograd.grad([energy.sum()], [im_neg_kl], create_graph=True)[0]

            im_neg_kl = im_neg_kl - cyc_lr * im_grad[:n]
            im_neg_kl = torch.clamp(im_neg_kl, 0, 1)
        else:
            im_neg = im_neg - cyc_lr * im_grad

        im_neg = im_neg.detach()

        if sample:
            im_negs_samples.append(im_neg)

        im_neg = torch.clamp(im_neg, 0, 1)

    if sample:
        return im_neg, im_neg_kl, im_negs_samples, np.abs(im_grad.detach().cpu().numpy()).mean()
    else:
        return im_neg, im_neg_kl, np.abs(im_grad.detach().cpu().numpy()).mean()