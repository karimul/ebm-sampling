from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision.datasets import CelebA

class CelebADataset(Dataset):
    def __init__(
            self,
            FLAGS,
            split='train',
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
            # transform = transforms.ToTensor()
            transform = transforms.Compose([
                # resize
                transforms.Resize(32),
                # center-crop
                transforms.CenterCrop(32),
                # to-tensor
                transforms.ToTensor()
            ])

        self.data = CelebA(
            "./data/celeba",
            transform=transform,
            split=split,
            download=True)
        self.one_hot_map = np.eye(10)
        self.noise = noise
        self.rescale = rescale
        self.FLAGS = FLAGS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        FLAGS = self.FLAGS
        
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
        # return torch.Tensor(im), label