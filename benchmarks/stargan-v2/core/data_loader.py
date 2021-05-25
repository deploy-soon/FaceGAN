"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
from os.path import join as pjoin
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)

class CelebaMultiLabelDataset(data.Dataset):

    def __init__(self, root, labels, transform=None):
        """
        LABEL LIST
        5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes
        Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry
        Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee
        Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open
        Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose
        Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair
        Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick
        Wearing_Necklace Wearing_Necktie Young
        """
        self.samples.self.targets = self._make_dataset(root)
        self.transform = transform
        self.labels = labels

        _mapper = {}
        with open("expr/CelebAMask-HQ-attribute-anno.txt", "r") as fin:
            num = fin.readline()
            headers = fin.readline().split()
            for label in labels:
                assert label in headers

            for line in fin.readlines():
                line = line.split()
                file_name, labels = line[0], line[1:]
                assert len(labels) == len(headers)
                for header, label in zip(headers, labels):
                    if label == '1' and header in self.labels:
                        _mapper.setdefault(file_name, []).append(labels.index(header))

        self.celeba_mapper = {}
        with open("expr/CelebA-HQ-to-CelebA-mapping.txt", "r") as fin:
            header = fin.readline()
            count = 0
            for line in fin.readlines():
                _, _, file_name = line.split()
                self.celeba_mapper[file_name] = _mapper[f"{count}.jpg"]
                count += 1

        self.samples, self.targets = self._make_dataset(root)

    def _make_dataset(self, root):
        #TODO: loading dataset from origin Celeba, currently from stargan parsed dataset
        images, labels = [], []
        for domain in ["male", "female"]:
            for fname in os.listdir(pjoin(root, domain)):
                if len(self.celeba_mapper[fname]) < 2:
                    continue
                images.append(pjoin(root, domain, fname))
                labels.append(self.celeba_mapper[fname])
        return images, labels

    def __getitem__(self, index):
        fname = self.samples[index]
        labels = self.targets[index]
        label1, label2 = random.sample(labels, 2)
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label1, label2

    def __len__(self):
        return len(self.targets)


class CelebaMultiLabelRefDataset(CelebaMultiLabelDataset):

    def __init__(self, root, labels, transform=None):
        super(CelebaMultiLabelRefDataset, self).__init__()

    def _make_dataset(self, root):
        _images, _labels = [], []
        for domain in ["male", "female"]:
            for fname in os.listdir(pjoin(root, domain)):
                if len(self.celeba_mapper[fname]) < 2:
                    continue
                _images.append(pjoin(root, domain, fname))
                _labels.append(self.celeba_mapper[fname])
        fnames1, fnames2, labels = [], [], []
        for label1 in range(len(self.labels)):
            for label2 in range(label1 + 1, len(self.labels)):

                _fnames = []
                for image, label in zip(_images, _labels):
                    if label1 in label and label2 in lable:
                        _fnames.append(image)
                fnames1 += len(_fnames)
                fnames2 += random.sample(_fnames, len(_fnames))
                labels += [(label1, label2)] * len(_fnames)

        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label1, label2 = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label1, label2


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
