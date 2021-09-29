import torch.nn as nn
from models.network import swish, CondResBlock

class MNISTModel(nn.Module):
    def __init__(self, args):
        super(MNISTModel, self).__init__()
        self.act = swish
        # self.relu = torch.nn.ReLU(inplace=True)

        self.args = args
        self.filter_dim = args.filter_dim
        self.init_main_model()
        self.init_label_map()
        self.filter_dim = args.filter_dim

        # self.act = self.relu
        self.cond = args.cond
        self.sigmoid = args.sigmoid


    def init_main_model(self):
        args = self.args
        filter_dim = self.filter_dim
        im_size = 28
        self.conv1 = nn.Conv2d(1, filter_dim, kernel_size=3, stride=1, padding=1)
        self.res1 = CondResBlock(args, filters=filter_dim, latent_dim=1, im_size=im_size)
        self.res2 = CondResBlock(args, filters=2*filter_dim, latent_dim=1, im_size=im_size)

        self.res3 = CondResBlock(args, filters=4*filter_dim, latent_dim=1, im_size=im_size)
        self.energy_map = nn.Linear(filter_dim*8, 1)


    def init_label_map(self):
        args = self.args

        self.map_fc1 = nn.Linear(10, 256)
        self.map_fc2 = nn.Linear(256, 256)

    def main_model(self, x, latent):
        x = x.view(-1, 1, 28, 28)
        x = self.act(self.conv1(x))
        x = self.res1(x, latent)
        x = self.res2(x, latent)
        x = self.res3(x, latent)
        x = self.act(x)
        x = x.mean(dim=2).mean(dim=2)
        energy = self.energy_map(x)

        return energy

    def label_map(self, latent):
        x = self.act(self.map_fc1(latent))
        x = self.map_fc2(x)

        return x

    def forward(self, x, latent):
        args = self.args
        x = x.view(x.size(0), -1)

        if self.cond:
            latent = self.label_map(latent)
        else:
            latent = None

        energy = self.main_model(x, latent)

        return energy