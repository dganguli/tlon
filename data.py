from functools import reduce

import numpy as np
import torch
import torchvision
from six.moves import urllib
from torch.utils.data import Dataset

from cgan import Generator


def load_mnist(save_path, batch_size_train=64, batch_size_test=1000):
    # this is a temporary work around to pytorch throwing 403s
    # see: https://github.com/pytorch/vision/issues/1938
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(save_path,
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize([0.5], [0.5])
                                   ])
                                   ),
        batch_size=batch_size_train,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(save_path, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize([0.5], [0.5])
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader


def load_synthetic_mnist(model_path, batch_size_train=64, batch_size_test=1000):
    train_loader = torch.utils.data.DataLoader(
        SyntheticMNISTDataset(model_path,
                              num_examples_per_target=6000,
                              seed=0
                              ),
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=0)

    test_loader = torch.utils.data.DataLoader(
        SyntheticMNISTDataset(model_path,
                              num_examples_per_target=1000,
                              seed=1
                              ),
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=0)

    return train_loader, test_loader


class SyntheticMNISTDataset(Dataset):
    def __init__(self,
                 model_path,
                 latent_dim=100,
                 img_size=(1, 28, 28),
                 num_examples_per_target=10,
                 seed=0
                 ):
        self.num_targets = 10
        self.num_examples_per_target = num_examples_per_target
        self.cgan = Generator(latent_dim, img_size).from_save_dict(model_path)

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.cgan.cuda()

        np.random.seed(seed)
        self.z = np.random.normal(0, 1, (self.num_targets * self.num_examples_per_target, latent_dim))
        self.targets = np.array(
            reduce(lambda x, y: x + y, [[d] * self.num_examples_per_target for d in range(self.num_targets)]))

    def __len__(self):
        return self.num_targets * self.num_examples_per_target

    def __getitem__(self, index):
        target = np.expand_dims(np.array(self.targets[index]), axis=0)
        z = np.expand_dims(self.z[index, :], axis=0)
        img = self.cgan.forward_numpy(z, target)
        img = img.squeeze(axis=0)

        return img, target[0]
