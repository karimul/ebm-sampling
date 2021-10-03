import torch
import numpy as np
from autograd.numpy import sqrt

def gen_image_resgld(label, FLAGS, model, im_neg, num_steps, sample=False):

    im_noise = torch.randn_like(im_neg).detach()

    T_multiply=0.9
    T = 0.9
    var=0.1
    resgld_beta_high = im_neg
    resgld_beta_low = im_neg
    swaps = 0

    noise_scale = sqrt(2e-6 * FLAGS.step_lr * T)

    print("noise_scale : ", noise_scale)
    print("noise_scale * T_multiply: ", noise_scale* T_multiply)

    im_negs_samples = []

    for i in range(num_steps):
        im_noise.normal_()

        resgld_beta_low = resgld_beta_low + noise_scale * im_noise
        resgld_beta_high = resgld_beta_high + noise_scale * T_multiply * im_noise

        resgld_beta_high.requires_grad_(requires_grad=True)
        energy_high = model.forward(resgld_beta_high, label)

        resgld_beta_low.requires_grad_(requires_grad=True)
        energy_low = model.forward(resgld_beta_low, label)

        im_grad_low = torch.autograd.grad([energy_low.sum()], [resgld_beta_low])[0]
        im_grad_high = torch.autograd.grad([energy_high.sum()], [resgld_beta_high])[0]
      
        if i == num_steps - 1:
            im_neg_orig = resgld_beta_low
            resgld_beta_low = resgld_beta_low - FLAGS.step_lr * im_grad_low 
            resgld_beta_high = resgld_beta_high - FLAGS.step_lr * im_grad_high 

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

                im_neg_kl = im_neg_kl - FLAGS.step_lr * im_grad[:n]
                im_neg_kl = torch.clamp(im_neg_kl, 0, 1)
        else:
            resgld_beta_low = resgld_beta_low - FLAGS.step_lr * im_grad_low
            resgld_beta_high = resgld_beta_high - FLAGS.step_lr * im_grad_high * T_multiply

        dT = 1 / T - 1 / (T * T_multiply)
        swap_rate = torch.exp(dT * (energy_low - energy_high - dT * var))
        intensity_r = 0.1
        # print("swap_rate", swap_rate)
        swap_rate = swap_rate.mean().item()
        print("swap_rate", swap_rate)
        random = np.random.uniform(0, 1)
        print("random", random)
        if random < intensity_r * swap_rate:
            resgld_beta_high, resgld_beta_low = resgld_beta_low, resgld_beta_high
            swaps += 1
            print("swaps : ", swaps)

        im_neg = resgld_beta_low.detach()

        if sample:
            im_negs_samples.append(im_neg)

        im_neg = torch.clamp(im_neg, 0, 1)

    if sample:
        return im_neg, im_neg_kl, im_negs_samples, np.abs(im_grad_low.detach().cpu().numpy()).mean()
    else:
        return im_neg, im_neg_kl, np.abs(im_grad_low.detach().cpu().numpy()).mean()