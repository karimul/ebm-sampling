import torch.nn as nn
from models.network import swish, CondResBlock, Downsample, Self_Attn
import torch
import torch.nn.functional as F

class ResNetModel(nn.Module):
    def __init__(self, args):
        super(ResNetModel, self).__init__()
        self.act = swish

        self.args = args
        self.spec_norm = args.spec_norm
        self.norm = args.norm
        self.init_main_model()

        if args.multiscale:
            self.init_mid_model()
            self.init_small_model()

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = Downsample(channels=3)

        self.cond = args.cond

    def init_main_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

        self.res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.res_3a = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_3b = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.res_4a = CondResBlock(args, filters=4*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_4b = CondResBlock(args, filters=4*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.self_attn = Self_Attn(2 * filter_dim, self.act)

        self.energy_map = nn.Linear(filter_dim*8, 1)

    def init_mid_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.mid_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.mid_res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.mid_res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

        self.mid_res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.mid_res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.mid_res_3a = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.mid_res_3b = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.mid_energy_map = nn.Linear(filter_dim*4, 1)
        self.avg_pool = Downsample(channels=3)

    def init_small_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.small_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.small_res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.small_res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

        self.small_res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.small_res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.small_energy_map = nn.Linear(filter_dim*2, 1)

    def main_model(self, x, latent, compute_feat=False):
        x = self.act(self.conv1(x))

        x = self.res_1a(x, latent)
        x = self.res_1b(x, latent)

        x = self.res_2a(x, latent)
        x = self.res_2b(x, latent)

        if self.args.self_attn:
            x, _ = self.self_attn(x)

        x = self.res_3a(x, latent)
        x = self.res_3b(x, latent)

        x = self.res_4a(x, latent)
        x = self.res_4b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        if compute_feat:
            return x

        x = x.view(x.size(0), -1)
        energy = self.energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        if self.args.sigmoid:
            energy = F.sigmoid(energy)

        return energy

    def mid_model(self, x, latent):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.mid_conv1(x))

        x = self.mid_res_1a(x, latent)
        x = self.mid_res_1b(x, latent)

        x = self.mid_res_2a(x, latent)
        x = self.mid_res_2b(x, latent)

        x = self.mid_res_3a(x, latent)
        x = self.mid_res_3b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        energy = self.mid_energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        if self.args.sigmoid:
            energy = F.sigmoid(energy)

        return energy

    def small_model(self, x, latent):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.small_conv1(x))

        x = self.small_res_1a(x, latent)
        x = self.small_res_1b(x, latent)

        x = self.small_res_2a(x, latent)
        x = self.small_res_2b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        energy = self.small_energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        if self.args.sigmoid:
            energy = F.sigmoid(energy)

        return energy

    def forward(self, x, latent):
        args = self.args

        if self.cond:
            latent = self.label_map(latent)
        else:
            latent = None

        energy = self.main_model(x, latent)

        if args.multiscale:
            large_energy = energy
            mid_energy = self.mid_model(x, latent)
            small_energy = self.small_model(x, latent)

            # Add a seperate energy penalizing the different energies from each model
            energy = torch.cat([small_energy, mid_energy, large_energy], dim=-1)

        return energy

    def compute_feat(self, x, latent):
        return self.main_model(x, None, compute_feat=True)