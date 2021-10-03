import torch
import numpy as np
from autograd.numpy.random import normal

def stochastic_f(energy): 
    return energy.detach().cpu().numpy() + 0.32*normal(size=1)

def gen_image_csgld(label, FLAGS, model, im_neg, num_steps, sample=False):
    im_noise = torch.randn_like(im_neg).detach()

    im_negs_samples = []

    parts = 100
    Gcum = np.array(range(parts, 0, -1)) * 1.0 / sum(range(parts, 0, -1))
    J = parts - 1
    bouncy_move = 0
    grad_mul = 1.
    zeta = 0.75
    T = 1
    decay_lr = 100.0

    for i in range(num_steps):
        im_noise.normal_()

        if FLAGS.anneal:
            im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * im_noise
        else:
            im_neg = im_neg + 0.001 * im_noise

        im_neg.requires_grad_(requires_grad=True)
        energy = model.forward(im_neg, label)
        # print("energy : ", energy)
        lower_bound, upper_bound = np.min(energy.detach().cpu().numpy()) - 1, np.max(energy.detach().cpu().numpy()) + 1
        partition=[lower_bound, upper_bound]

        if FLAGS.all_step:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        else:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg])[0]

        if i == num_steps - 1:
            im_neg_orig = im_neg
            im_neg = im_neg - FLAGS.step_lr * grad_mul * im_grad

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

            im_neg_kl = im_neg_kl - FLAGS.step_lr * grad_mul * im_grad[:n]
            im_neg_kl = torch.clamp(im_neg_kl, 0, 1)
        else:
            im_neg = im_neg - FLAGS.step_lr * grad_mul * im_grad

        print("\n grad_mul: ", grad_mul)
        div_f = (partition[1] - partition[0]) / parts
        grad_mul = 1 + zeta * T * (np.log(Gcum[J]) - np.log(Gcum[J-1])) / div_f
      
        J = (min(max(int((stochastic_f(energy).mean() - partition[0]) / div_f + 1), 1), parts - 1))
        step_size = min(decay_lr, 10./(i**0.8+100))
        Gcum[:J] = Gcum[:J] + step_size * Gcum[J]**zeta * (-Gcum[:J])
        Gcum[J] = Gcum[J] + step_size * Gcum[J]**zeta * (1 - Gcum[J])
        Gcum[(J+1):] = Gcum[(J+1):] + step_size * Gcum[J]**zeta * (-Gcum[(J+1):])

        if grad_mul < 0:
            bouncy_move = bouncy_move + 1
            print("\n bouncy_move : ", bouncy_move)

        im_neg = im_neg.detach()

        if sample:
            im_negs_samples.append(im_neg)

        im_neg = torch.clamp(im_neg, 0, 1)

    if sample:
        return im_neg, im_neg_kl, im_negs_samples, np.abs(im_grad.detach().cpu().numpy()).mean()
    else:
        return im_neg, im_neg_kl, np.abs(im_grad.detach().cpu().numpy()).mean()