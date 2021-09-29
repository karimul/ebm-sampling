from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torchvision
import torch

class Mnist(Dataset):
    def __init__(self, train=True, rescale=1.0):
        self.data = torchvision.datasets.MNIST(
            "data/mnist",
            transform=transforms.ToTensor(),
            download=True, train=train)
        self.labels = np.eye(10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        im, label = self.data[index]
        label = self.labels[label]
        im = im.squeeze()
        im = im.numpy() / 256 * 255 + np.random.uniform(0, 1. / 256, (28, 28))
        im = np.clip(im, 0, 1)
        s = 28
        im_corrupt = np.random.uniform(0, 1, (s, s, 1))
        im = im[:, :, None]

        return torch.Tensor(im_corrupt), torch.Tensor(im), label