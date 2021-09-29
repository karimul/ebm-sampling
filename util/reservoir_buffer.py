from utils import GaussianBlur
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

class ReservoirBuffer(object):
    def __init__(self, size, transform, dataset):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.n = 0

        def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            color_distort = transforms.Compose([
                rnd_color_jitter,
                rnd_gray])
            return color_distort

        if dataset in ("cifar10", "celeba", "cats"):
            im_size = 32
        elif dataset == "continual":
            im_size = 64
        elif dataset == "celeba":
            im_size = 128
        elif dataset == "object":
            im_size = 128
        elif dataset == "mnist":
            im_size = 28
        elif dataset == "moving_mnist":
            im_size = 28
        elif dataset == "imagenet":
            im_size = 128
        elif dataset == "lsun":
            im_size = 128
        elif dataset == "stl":
            im_size = 48
        else:
            assert False

        color_transform = get_color_distortion(0.5)
        self.dataset = dataset

        if transform:
            if dataset in ("cifar10", "celeba", "cats"):
                color_transform = get_color_distortion(1.0)
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
                # self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.03, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "continual":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "celeba":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "imagenet":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=11), transforms.ToTensor()])
            elif dataset == "lsun":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "stl":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.04, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=11), transforms.ToTensor()])
            elif dataset == "object":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
            elif dataset == "mnist":
                self.transform = None
            elif dataset == "moving_mnist":
                self.transform = None
            else:
                assert False
        else:
            self.transform = None

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
            self.n = self.n + ims.shape[0]
        else:
            for im in ims:
                self.n = self.n + 1
                ix = random.randint(0, self.n - 1)

                if ix < len(self._storage):
                    self._storage[ix] = im

        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize


    def _encode_sample(self, idxes, no_transform=False, downsample=False):
        ims = []
        for i in idxes:
            im = self._storage[i]

            if self.dataset != "mnist":
                if (self.transform is not None) and (not no_transform):
                    im = im.transpose((1, 2, 0))
                    im = np.array(self.transform(Image.fromarray(im)))

            im = im * 255

            ims.append(im)
        return np.array(ims)

    def sample(self, batch_size, no_transform=False, downsample=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes, no_transform=no_transform, downsample=downsample), idxes