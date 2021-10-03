import os
from torch.utils.tensorboard import SummaryWriter
from util.replay_buffer import ReplayBuffer
from util.reservoir_buffer import ReservoirBuffer
import time
import torch
from tqdm import tqdm
import numpy as np
from sampling.asgld import gen_image_asgld
from sampling.csgld import gen_image_csgld
from sampling.cycsgld import gen_image_cycsgld
from sampling.psgld import gen_image_psgld
from sampling.resgld import gen_image_resgld
from sampling.sgld import gen_image
from geomloss import SamplesLoss
import timeit
from torch.nn.utils import clip_grad_norm_
from util.utils import rescale_im
from torchmetrics import IS, FID
import os.path as osp
from dataset.celaba_dataset import CelebADataset
from dataset.cifar10_dataset import Cifar10
from dataset.mnist_dataset import Mnist
from models.mnist_model import MNISTModel
from models.resnet_model import ResNetModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from azureml.core import Run
from torchvision.utils import make_grid

run = Run.get_context()

def compress_x_mod(x_mod):
    x_mod = (255 * np.clip(x_mod, 0, 1)).astype(np.uint8)
    return x_mod


def decompress_x_mod(x_mod):
    x_mod = x_mod / 256  + \
        np.random.uniform(0, 1 / 256, x_mod.shape)
    return x_mod

def log_tensorboard(writer, data):
    batch_size = data["negative_samples"].shape[0]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(data["negative_samples"][:batch_size/2], nrow=8).permute(1, 2, 0))
    # img_grid = make_grid(data["negative_samples"], nrow=8).permute(1, 2, 0)
    # plt.imshow(img_grid) 

    img_name = "negative_examples_1_" +  str(data["iter"])
    run.log_image(name=img_name, plot=fig)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(data["negative_samples"][batch_size/2:], nrow=8).permute(1, 2, 0))
    img_name = "negative_examples_2_" +  str(data["iter"])
    run.log_image(name=img_name, plot=fig)

    run.log_row("IS", x=data["iter"], y=data["is_mean"])
    run.log_row("FID", x=data["iter"], y=data["fid"])
    run.log_row("energy different", x=data["iter"], y=data["e_diff"])

    writer.add_scalar("replay buffer length", data["length_replay_buffer"], )
    writer.add_scalar("repel loss", data["loss_repel"], data["iter"])
    writer.add_scalar("batch loss", data["loss"], data["iter"])
    writer.add_scalar("average loss", data["avg_loss"], data["iter"])
    writer.add_scalar("KL mean loss", data["kl_mean"], data["iter"])
    
    writer.add_scalar("FID", data["fid"], data["iter"])
    writer.add_scalar("IS mean", data["is_mean"], data["iter"])
    writer.add_scalar("IS std", data["is_std"], data["iter"])
    writer.add_scalar("SSIM", data["ssim"], data["iter"])

    writer.add_scalar("positive energy mean", data["e_pos"], data["iter"])
    writer.add_scalar("positive energy std", data["e_pos_std"], data["iter"])

    writer.add_scalar("negative energy mean", data["e_neg"], data["iter"])
    writer.add_scalar("negative energy std", data["e_neg_std"], data["iter"])

    writer.add_scalar("energy different", data["e_diff"], data["iter"])
    writer.add_scalar("x gradient", data["x_grad"], data["iter"])

    writer.add_images("positive examples", data["positive_samples"], data["iter"])
    writer.add_images("negative examples", data["negative_samples"], data["iter"])

def train(model, optimizer, dataloader, FLAGS):
    writer = SummaryWriter(log_dir=FLAGS.logdir)
    inception = IS().to(FLAGS.gpu, non_blocking=True)
    fid = FID(feature=2048).to(FLAGS.gpu, non_blocking=True)

    if FLAGS.replay_batch:
        if FLAGS.reservoir:
            replay_buffer = ReservoirBuffer(FLAGS.buffer_size, FLAGS.transform, FLAGS.dataset)
        else:
            replay_buffer = ReplayBuffer(FLAGS.buffer_size, FLAGS.transform, FLAGS.dataset)
    dist_sinkhorn = SamplesLoss('sinkhorn')
    itr = FLAGS.resume_iter
    im_neg = None
    gd_steps = 1

    optimizer.zero_grad()

    num_steps = FLAGS.num_steps

    for epoch in range(FLAGS.epoch_num):
        print("epoch : ", epoch)
        tock = time.time()
        average_loss = 0.0
        for data_corrupt, data, label in tqdm(dataloader):
            label = label.float().to(FLAGS.gpu, non_blocking=True)
            data = data.permute(0, 3, 1, 2).float().contiguous()
            
            # Generate samples to evaluate inception score
            if itr % FLAGS.save_interval == 0:
                if FLAGS.dataset in ("cifar10", "celeba", "cats"):
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (128, 32, 32, 3)))
                    repeat = 128 // FLAGS.batch_size + 1
                    label = torch.cat([label] * repeat, axis=0)
                    label = label[:128]
                elif FLAGS.dataset == "celebahq":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (data.shape[0], 128, 128, 3)))
                    label = label[:data.shape[0]]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "stl":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (32, 48, 48, 3)))
                    label = label[:32]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "lsun":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (32, 128, 128, 3)))
                    label = label[:32]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "imagenet":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (32, 128, 128, 3)))
                    label = label[:32]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "object":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (32, 128, 128, 3)))
                    label = label[:32]
                    data_corrupt = data_corrupt[:label.shape[0]]
                elif FLAGS.dataset == "mnist":
                    data_corrupt = torch.Tensor(np.random.uniform(0.0, 1.0, (128, 28, 28, 1)))
                    label = label[:128]
                    data_corrupt = data_corrupt[:label.shape[0]]
                else:
                    assert False
            
            data_corrupt = torch.Tensor(data_corrupt.float()).permute(0, 3, 1, 2).float().contiguous()
            data = data.to(FLAGS.gpu, non_blocking=True)
            data_corrupt = data_corrupt.to(FLAGS.gpu, non_blocking=True)
            
            if FLAGS.replay_batch and len(replay_buffer) >= FLAGS.batch_size:
                replay_batch, idxs = replay_buffer.sample(data_corrupt.size(0))
                replay_batch = decompress_x_mod(replay_batch)
                replay_mask = (
                    np.random.uniform(
                        0,
                        1,
                        data_corrupt.size(0)) > 0.001)
                data_corrupt[replay_mask] = torch.Tensor(replay_batch[replay_mask]).to(FLAGS.gpu, non_blocking=True)
            else:
                idxs = None

            if FLAGS.sampler == "psgld":
                if itr % FLAGS.save_interval == 0:
                    im_neg, im_neg_kl, im_samples, x_grad = gen_image_psgld(label, FLAGS, model, data_corrupt, num_steps, sample=True)
                else:
                    im_neg, im_neg_kl, x_grad = gen_image_psgld(label, FLAGS, model, data_corrupt, num_steps)       
            elif FLAGS.sampler == "asgld":
                if itr % FLAGS.save_interval == 0:
                    im_neg, im_neg_kl, im_samples, x_grad = gen_image_asgld(label, FLAGS, model, data_corrupt, num_steps, sample=True)
                else:
                    im_neg, im_neg_kl, x_grad = gen_image_asgld(label, FLAGS, model, data_corrupt, num_steps)
            elif FLAGS.sampler == "sgld":
                if itr % FLAGS.save_interval == 0:
                    im_neg, im_neg_kl, im_samples, x_grad = gen_image(label, FLAGS, model, data_corrupt, num_steps, sample=True)
                else:
                    im_neg, im_neg_kl, x_grad = gen_image(label, FLAGS, model, data_corrupt, num_steps)
            elif FLAGS.sampler == "cycsgld":
                if itr % FLAGS.save_interval == 0:
                    im_neg, im_neg_kl, im_samples, x_grad = gen_image_cycsgld(label, FLAGS, model, data_corrupt, num_steps, sample=True)
                else:
                    im_neg, im_neg_kl, x_grad = gen_image_cycsgld(label, FLAGS, model, data_corrupt, num_steps)
            elif FLAGS.sampler == "resgld":
                if itr % FLAGS.save_interval == 0:
                    im_neg, im_neg_kl, im_samples, x_grad = gen_image_resgld(label, FLAGS, model, data_corrupt, num_steps, sample=True)
                else:
                    im_neg, im_neg_kl, x_grad = gen_image_resgld(label, FLAGS, model, data_corrupt, num_steps)
            elif FLAGS.sampler == "csgld":
                if itr % FLAGS.save_interval == 0:
                    im_neg, im_neg_kl, im_samples, x_grad = gen_image_csgld(label, FLAGS, model, data_corrupt, num_steps, sample=True)
                else:
                    im_neg, im_neg_kl, x_grad = gen_image_csgld(label, FLAGS, model, data_corrupt, num_steps)
            else:
                assert False
            
            data_corrupt = None
            energy_pos = model.forward(data, label[:data.size(0)])
            energy_neg = model.forward(im_neg, label)
            
            if FLAGS.replay_batch and (im_neg is not None):
                replay_buffer.add(compress_x_mod(im_neg.detach().cpu().numpy()))

            loss = energy_pos.mean() - energy_neg.mean() 
            loss = loss  + (torch.pow(energy_pos, 2).mean() + torch.pow(energy_neg, 2).mean())

            if FLAGS.kl:
                model.requires_grad_(False)
                loss_kl = model.forward(im_neg_kl, label)
                model.requires_grad_(True)
                loss = loss + FLAGS.kl_coeff * loss_kl.mean()

                if FLAGS.repel_im:
                    start = timeit.timeit()
                    bs = im_neg_kl.size(0)

                    if FLAGS.dataset in ["celebahq", "imagenet", "object", "lsun", "stl"]:
                        im_neg_kl = im_neg_kl[:, :, :, :].contiguous()

                    im_flat = torch.clamp(im_neg_kl.view(bs, -1), 0, 1)

                    if FLAGS.dataset in ("cifar10", "celeba", "cats"):
                        if len(replay_buffer) > 1000:
                            compare_batch, idxs = replay_buffer.sample(100, no_transform=False)
                            compare_batch = decompress_x_mod(compare_batch)
                            compare_batch = torch.Tensor(compare_batch).to(FLAGS.gpu, non_blocking=True)
                            compare_flat = compare_batch.view(100, -1)

                            if FLAGS.entropy == 'kl':
                                dist_matrix = torch.norm(im_flat[:, None, :] - compare_flat[None, :, :], p=2, dim=-1)
                                loss_repel = torch.log(dist_matrix.min(dim=1)[0]).mean()
                                # loss_repel = kldiv(im_flat, compare_flat)
                                loss = loss - 0.3 * loss_repel
                            elif FLAGS.entropy == 'sinkhorn':
                                dist_matrix = dist_sinkhorn(im_flat, compare_flat)
                                loss_repel = torch.log(dist_matrix).sum()
                                loss = loss - 0.03 * loss_repel
                            else:
                                assert False                     
                        else:
                            loss_repel = torch.zeros(1)
                        
                        # loss = loss - 0.3 * loss_repel
                    else:
                        if len(replay_buffer) > 1000:
                            compare_batch, idxs = replay_buffer.sample(100, no_transform=False, downsample=True)
                            compare_batch = decompress_x_mod(compare_batch)
                            compare_batch = torch.Tensor(compare_batch).to(FLAGS.gpu, non_blocking=True)
                            compare_flat = compare_batch.view(100, -1)
                            
                            if FLAGS.entropy == 'kl':
                                dist_matrix = torch.norm(im_flat[:, None, :] - compare_flat[None, :, :], p=2, dim=-1)
                                loss_repel = torch.log(dist_matrix.min(dim=1)[0]).mean()
                                # loss_repel = kldiv(im_flat, compare_flat)
                            elif FLAGS.entropy == 'sinkhorn':
                                dist_matrix = dist_sinkhorn(im_flat, compare_flat)
                                loss_repel = torch.log(dist_matrix).sum()
                            else:
                                assert False
                        else:
                            loss_repel = torch.zeros(1).to(FLAGS.gpu, non_blocking=True)

                        if FLAGS.entropy == 'kl':
                            loss = loss - 0.3 * loss_repel  
                        elif FLAGS.entropy == 'sinkhorn':
                            loss = loss - 0.03 * loss_repel
                        else:
                            assert False

                    end = timeit.timeit()
                else:
                    loss_repel = torch.zeros(1)

            else:
                loss_kl = torch.zeros(1)
                loss_repel = torch.zeros(1)

            if FLAGS.log_grad and len(replay_buffer) > 1000:
                loss_kl = loss_kl - 0.1 * loss_repel
                loss_kl = loss_kl.mean()
                loss_ml = energy_pos.mean() - energy_neg.mean()

                loss_ml.backward(retain_graph=True)
                ele = []

                for param in model.parameters():
                    if param.grad is not None:
                        ele.append(torch.norm(param.grad.data))

                ele = torch.stack(ele, dim=0)
                ml_grad = torch.mean(ele)
                model.zero_grad()

                loss_kl.backward(retain_graph=True) 
                ele = []

                for param in model.parameters():
                    if param.grad is not None:
                        ele.append(torch.norm(param.grad.data))

                ele = torch.stack(ele, dim=0)
                kl_grad = torch.mean(ele)
                model.zero_grad()

            else:
                ml_grad = None
                kl_grad = None

            loss.backward()

            clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            optimizer.zero_grad()

            # ema_model(models, models_ema)

            if torch.isnan(energy_pos.mean()):
                assert False

            if torch.abs(energy_pos.mean()) > 10.0:
                assert False
            
            average_loss += (loss - average_loss) / (itr + 1)
            if itr % FLAGS.log_interval == 0:
                tick = time.time()

                kvs = {}
                kvs['e_pos'] = energy_pos.mean().item()
                kvs['e_pos_std'] = energy_pos.std().item()
                kvs['e_neg'] = energy_neg.mean().item()
                kvs['kl_mean'] = loss_kl.mean().item()
                kvs['loss_repel'] = loss_repel.mean().item()
                kvs['loss'] = loss
                kvs['avg_loss'] = average_loss
                kvs['e_neg_std'] = energy_neg.std().item()
                kvs['e_diff'] = kvs['e_pos'] - kvs['e_neg']
                # kvs['x_grad'] = np.abs(x_grad.detach().cpu().numpy()).mean()
                kvs['x_grad'] = x_grad
                kvs['iter'] = itr
                # kvs['hmc_loss'] = hmc_loss.item()
                kvs['num_steps'] = num_steps
                # kvs['t_diff'] = tick - tock
                kvs['positive_samples'] = data.cpu().detach()
                kvs['negative_samples'] = im_neg.cpu().detach()

                real = data.cpu().detach()
                fake = im_neg.cpu().detach()
                data = None
                im_neg = None
                if real.shape[1] == 1:
                    # print("channel 1")
                    real = torch.cat((real, real, real), dim=1)
                    fake = torch.cat((fake, fake, fake), dim=1)
                real = torch.from_numpy(rescale_im(real.cpu().numpy())).to(FLAGS.gpu, non_blocking=True)
                fake = torch.from_numpy(rescale_im(fake.cpu().numpy())).to(FLAGS.gpu, non_blocking=True)
                # print("real shape = ", real.shape)
                # print("campute IS")
                inception.update(fake)
                inception_mean, inception_std = inception.compute()
                # print("campute FID")
                fid.update(real, real=True)
                fid.update(fake, real=False)
                fid_val = fid.compute()
                real = None
                fake = None
                ssim_value = 0
                kvs['fid'] = fid_val.item()
                kvs['is_mean'] = inception_mean.item()
                kvs['is_std'] = inception_std.item()
                kvs['ssim'] = ssim_value

                if FLAGS.replay_batch:
                    kvs['length_replay_buffer'] = len(replay_buffer)

                log_tensorboard(writer, kvs)
                tock = tick

            if itr % FLAGS.save_interval == 0 and (FLAGS.save_interval != 0):
                model_path = osp.join(FLAGS.logdir, "model_{}.pth".format(itr))
                ckpt = {'optimizer_state_dict': optimizer.state_dict(),
                            'FLAGS': FLAGS}

                for i in range(FLAGS.ensembles):
                    ckpt['model_state_dict_{}'.format(i)] = model.state_dict()
                    # ckpt['ema_model_state_dict_{}'.format(i)] = model.state_dict()

                torch.save(ckpt, model_path)


            itr += 1
            
def main_single(FLAGS):
    print("Values of args: ", FLAGS)

    if FLAGS.dataset == "cifar10":
        train_dataset = Cifar10(FLAGS)
        # valid_dataset = Cifar10(FLAGS, split='valid', augment=False)
        # test_dataset = Cifar10(FLAGS, split='test', augment=False)
    elif FLAGS.dataset == "celeba":
        train_dataset = CelebADataset(FLAGS)
        # valid_dataset = CelebADataset(FLAGS, train=False, augment=False)
        # test_dataset = CelebADataset(FLAGS, train=False, augment=False)
    elif FLAGS.dataset == "mnist":
        train_dataset = Mnist(FLAGS)
        # valid_dataset = Mnist(train=False)
        # test_dataset = Mnist(train=False)
    else:
        assert False

    train_dataloader = DataLoader(train_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
    # valid_dataloader = DataLoader(valid_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
    # test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
    
    if FLAGS.resume_iter != 0:
        FLAGS_OLD = FLAGS
        model_path = osp.join(FLAGS.logdir, "model_{}.pth".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path)
        FLAGS = checkpoint['FLAGS']

        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS_OLD = None

    if FLAGS.dataset in ("cifar10", "celeba", "cats"):
        model_fn = ResNetModel
    elif FLAGS.dataset == "stl":
        model_fn = ResNetModel
    elif FLAGS.dataset == "mnist":
        model_fn = MNISTModel
    else:
        assert False

    model = model_fn(FLAGS).train()
    # models_ema = model_fn(FLAGS).train()

    if torch.cuda.is_available():
        model = model.to(FLAGS.gpu)

    optimizer = Adam(model.parameters(), lr=FLAGS.lr, betas=(0.0, 0.9), eps=1e-8)

    # ema_model(models, models_ema, mu=0.0)

    it = FLAGS.resume_iter

    if not osp.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    checkpoint = None
    if FLAGS.resume_iter != 0:
        print("FLAGS.resume_iter:",FLAGS.resume_iter)
        model_path = osp.join(FLAGS.logdir, "model_{}.pth".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for i in range(FLAGS.ensembles):
            model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)])
            # model_ema.load_state_dict(checkpoint['ema_model_state_dict_{}'.format(i)])
 

    print("New Values of args: ", FLAGS)

    pytorch_total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Number of parameters for models", pytorch_total_params)

    train(model, optimizer, train_dataloader, FLAGS)