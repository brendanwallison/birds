from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from base import BaseDataLoader
from six.moves import urllib
from parse_config import ConfigParser

# downloads
import requests
import json
from collections import Counter 
import os
import errno
import csv
import numpy as np
import pandas as pd
import splitfolders
import pathlib    
import torchaudio
import torch

# Note: horizontal dimension = 2 * time_window * sample_rate // n_fft + 1
# vertical crop = n_fft // 2 + 1
class SpectrogramLoader(BaseDataLoader):
    def __init__(self, dataset=None, batch_size=128, shuffle=False, validation_split=0.0, weighted_sample = False, num_workers=1, data_dir="data/processed", training=True):
        self.dataset = dataset        
        self.data_dir = data_dir   
        if dataset is not None:
            self.vertical_crop = dataset.vertical_crop   
            self.horizontal_crop = dataset.horizontal_crop     
            if dataset.mode == 'xeno':
            # Stack of numpy melspecs -> one torch melspec
            #self.horizontal_crop=dataset.horizontal_crop - 1
                trsfm = transforms.Compose([
                    RandomImage(dataset.split_files, self.horizontal_crop),
                    #Superimpose(self.dataset, dataset.split_files, self.horizontal_crop),
                    NormalizeLabels(),
                    ThreeChannel(),
                    NumpyStackToTensors()
                #transforms.RandomCrop(size = (self.vertical_crop, self.horizontal_crop), pad_if_needed=True, padding_mode = 'constant')
            ])
            else:
                trsfm = transforms.Compose([
                    # RandomImage(),
                    ThreeChannel(),
                    AxisOrderChange(),
                    NumpyStackToTensors(),
                    Crop()
                    #transforms.ToTensor(),
                    #transforms.RandomCrop(size = (self.vertical_crop, self.horizontal_crop), pad_if_needed=True, padding_mode = 'constant')
                ])
            dataset.set_transform(trsfm)
        else:
            self.vertical_crop = 128     
            self.horizontal_crop = 281 
            dataset = datasets.DatasetFolder(root = self.data_dir, loader = self.default_loader, transform = trsfm, extensions=('.pickle'))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, weighted_sample, num_workers)


    # assumes we have used torch.save() or another pickle saver
    # on tensor-based spectrogram
    def default_loader(self, path):
        mel_specgram = torch.load(path)
        return mel_specgram.numpy()

class AddChannel(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, t):
        sample = t[0]
        label = t[1]
        new_sample = sample[:, :, None]
        return (new_sample, label)

class RandomImage(object):
    # Pick a random image from a stack
    def __init__(self, split_files, horizontal_crop = None):
        self.split_files = split_files
        self.horizontal_crop = horizontal_crop
    def __call__(self, t):
        sample = t[0]
        label = t[1]
        if self.split_files:
            choices = range(sample.shape[0])
            choice = np.random.choice(choices)
            new_sample = sample[choice]
        else:
            low = 0
            high = sample.shape[-1] - self.horizontal_crop - 1
            while high < 0:
                sample = np.hstack((sample, sample))
                high = sample.shape[-1] - self.horizontal_crop - 1
            offset = int(np.random.uniform(low=low, high=high, size = 1))
            new_sample = sample[..., offset: offset+self.horizontal_crop]
        return (new_sample, label)
        

class ThreeChannel(object):
    # Converts a stack of images to color
    def __call__(self, t):
        sample = t[0]
        label = t[1]
        sample = np.stack([sample, sample, sample])
        return (sample, label)

class NumpyStackToTensors(object):
    def __call__(self, t):
        sample = t[0]
        label = t[1]
        sample = [transforms.ToTensor()(sample[i]) for i in range(sample.shape[0])]
        sample = torch.stack(sample)
        return (torch.squeeze(sample), label)  

class AxisOrderChange(object):
    # Torch tensor transform expects:
    # HxWxC, from 0 to 255
    # Returns CxHxW
    def __call__(self, t):
        sample = t[0]
        label = t[1]
        sample = np.moveaxis(sample, 0, -1)
        return (sample, label)

class Crop(object):
    def __call__(self, t):
        sample = t[0]
        label = t[1]
        return (TF.crop(sample, top = 0, left = 0, height = 128, width = 201), label)

# Assumes one image file
class Superimpose(object):
    def __init__(self, dataset, split_files, horizontal_crop = None):
        self.dataset = dataset
        self.split_files = split_files
        self.horizontal_crop = horizontal_crop
    def __call__(self, t):
        sample = t[0]
        label = t[1]
        mix_idx = np.random.choice(len(self.dataset))
        mixer, mix_label = self.dataset.__getitem__(mix_idx, no_transform = True)
        mixer, mix_label = RandomImage(self.split_files, self.horizontal_crop)((sample, mix_label))
        w = self.weight()
        sample = sample + mixer*w
        label = label + mix_label*w
        return (sample, label)
    
    def weight(self):
        w = np.random.beta(1, 3)
        return w

# Assumes one image file
class NormalizeLabels(object):
    def __call__(self, t):
        sample = t[0]
        label = t[1]
        sum_of_rows = torch.sum(label)
        normalized_labels = label / sum_of_rows
        return (sample, normalized_labels)
    

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    #def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, weighted_sample = False, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # headers required for valid request to cloudflare-protected dataset
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
