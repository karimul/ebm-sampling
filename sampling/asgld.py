import torch
import numpy as np

def gen_image_asgld(label, FLAGS, model, im_neg, num_steps, sample=False):
    stepsize = 0.2
    noise_scale = np.sqrt(stepsize * 0.01)
    im_noise = torch.randn_like(im_neg).detach() * noise_scale

    im_negs_samples = []
    
    # Intialize mean and variance to zero
    mean = torch.zeros_like(im_neg.data)
    std = torch.zeros_like(im_neg.data)
    weight_decay = 5e-4
    v_noise=0.001
    momentum=0.9
    eps=1e-6
    for i in range(num_steps):
        # im_noise.normal_()
        # Getting mean,std at previous step
        old_mean = mean.clone()
        old_std = std.clone()

        im_noise = torch.normal(mean=old_mean, std=old_std)
        # updt = x_negative.data.add(v_noise,im_noise)

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

        # Updating mean
        mean = mean.mul(momentum).add(im_neg)
        
        # Updating std
        part_var1 = im_neg.add(-old_mean)
        part_var2 = im_neg.add(-mean)
        
        new_std = torch.pow(old_std,2).mul(momentum).addcmul(1,part_var1,part_var2).add(eps)                
        new_std = torch.pow(torch.abs_(new_std),1/2)
        std.add_(-1,std).add_(new_std)        

        if i == num_steps - 1:
            im_neg_orig = im_neg
            im_neg = im_neg - FLAGS.step_lr * im_grad

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