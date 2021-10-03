import torch
import numpy as np

def gen_image_psgld(label, FLAGS, model, im_neg, num_steps, sample=False):
    square_avg = torch.zeros_like(im_neg)
    im_negs_samples = []

    for i in range(num_steps):

        avg = square_avg.sqrt().add_(FLAGS.eps)
        im_noise = torch.normal(mean=0,std=avg)

        if FLAGS.anneal:
            im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * im_noise
        else:
            im_neg = im_neg + 0.001 * im_noise

        im_neg.requires_grad_(requires_grad=True)
        energy = model.forward(im_neg, label)

        if FLAGS.all_step:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        else:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg])[0]

        square_avg.mul_(FLAGS.momentum).addcmul_(1 - FLAGS.momentum, im_neg.data, im_neg.data)
        
        if i == num_steps - 1:
            im_neg_orig = im_neg
            im_neg = im_neg - FLAGS.step_lr * im_grad / avg

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
            im_neg = im_neg - FLAGS.step_lr * im_grad

        im_neg = im_neg.detach()

        if sample:
            im_negs_samples.append(im_neg)

        im_neg = torch.clamp(im_neg, 0, 1)

    if sample:
        return im_neg, im_neg_kl, im_negs_samples, np.abs(im_grad.detach().cpu().numpy()).mean()
    else:
        return im_neg, im_neg_kl, np.abs(im_grad.detach().cpu().numpy()).mean()