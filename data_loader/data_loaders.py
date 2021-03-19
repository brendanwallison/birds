from torchvision import datasets, transforms
from base import BaseDataLoader
from six.moves import urllib
from parse_config import ConfigParser

# download xeno imports
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
import librosa


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

class BirdsongDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, weighted_sample = False, num_workers=1, training=True, location='Mexico', num_birds=3, n_fft = 400, time_slice = 10, resample_rate = 22050):
        self.n_fft = n_fft
        self.time_slice = time_slice
        self.resample_rate = resample_rate
        self.min_waveform_width = time_slice * resample_rate
        # keep each fft bin (vertical); crop horizontal to correspond to desired timeslice
        self.horizontal_crop = 2 * time_slice * resample_rate / n_fft
        self.vertical_crop = n_fft // 2 + 1

        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(size = (self.vertical_crop, self.horizontal_crop), pad_if_needed=True, padding_mode = 'constant')
        ])
        
        # headers required for valid request to cloudflare-protected dataset
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        self.data_dir = data_dir
        self.dataset = datasets.DatasetFolder(root = self.data_dir, loader = self.default_loader, extensions=('.wav', '.mp3', '.WAV'))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, weighted_sample, num_workers)


    def default_loader(self, path):
        # TO-DO: more efficient to (randmoly) crop now 
        # using frame_offset and num_frames parameters of torchaudio.load
        # need to verify it is called every epoch, so that data augmentation still works      
        waveform, sample_rate = torchaudio.load(path, normalization = True)
        waveform = torch.mean(input = waveform, dim = 1) 
        needed_waveform_length = 
        repeats = len(waveform[1]) 
        mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)  # (channel, n_mels, time)
        return mel_specgram


