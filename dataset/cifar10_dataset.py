from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision

class Cifar10(Dataset):
    def __init__(
            self,
            FLAGS,
            train=True,
            full=False,
            augment=False,
            noise=True,
            rescale=1.0):

        if augment:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]

            transform = transforms.Compose(transform_list)
        else:
            transform = transforms.ToTensor()

        self.full = full
        self.data = torchvision.datasets.CIFAR10(
            FLAGS.datadir,
            transform=transform,
            train=train,
            download=FLAGS.download)
        self.test_data = torchvision.datasets.CIFAR10(
            FLAGS.datadir,
            transform=transform,
            train=False,
            download=FLAGS.download)
        self.one_hot_map = np.eye(10)
        self.noise = noise
        self.rescale = rescale
        self.FLAGS = FLAGS

    def __len__(self):

        if self.full:
            return len(self.data) + len(self.test_data)
        else:
            return len(self.data)

    def __getitem__(self, index):
        FLAGS = self.FLAGS
        if self.full:
            if index >= len(self.data):
                im, label = self.test_data[index - len(self.data)]
            else:
                im, label = self.data[index]
        else:
            im, label = self.data[index]

        im = np.transpose(im, (1, 2, 0)).numpy()
        image_size = 32
        label = self.one_hot_map[label]

        im = im * 255 / 256

        im = im * self.rescale + \
            np.random.uniform(0, 1 / 256., im.shape)

        # np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        im_corrupt = np.random.uniform(
            0.0, self.rescale, (image_size, image_size, 3))

        return torch.Tensor(im_corrupt), torch.Tensor(im), label