import os
import random
import numpy as np
from torch.utils.data import Dataset
from skimage.color import rgb2hsv, hsv2rgb
from scipy.misc import imresize
from torch.utils.data.sampler import WeightedRandomSampler


class Internal(Dataset):

    def __init__(self, files, patch_size=32, augment=False):
        self.files  = files
        self.patch_size = patch_size
        self.augment = augment
        random.shuffle(self.files)

    def get_weighted_random_sampler(self, num_samples):
        ''' Replace every label with the reciprocal of its frequency. '''
        # only consider genuine versus spoof
        labels = [0 if l==0  else 1 for f,l in self.files]
        # replace label with how often it occurs
        train_weights = np.bincount(labels)[labels]
        # print(train_weights)
        # turn occurance into weight and create the random sampler
        train_weights = 1.0 / train_weights
        print(num_samples)
        train_sampler = WeightedRandomSampler(train_weights, num_samples=int(num_samples))
        return train_sampler

    def __getitem__(self, idx):
        ''' Return a single preprocessed image. '''
        filename, label = self.files[idx]
        # load frame f as numpy memmap for partial loading
        try: mmap = np.load(filename, mmap_mode='r')
        except: print(filename)
                
        # extract a random patch from the memmap
        # print (mmap.shape)
        if self.patch_size is not None:
            h, w, c = mmap.shape
            h = random.randint(0, h - self.patch_size)
            w = random.randint(0, w - self.patch_size)
            rgb = np.array(mmap[h:h + self.patch_size, w:w + self.patch_size, :])
        else:
            rgb = np.array(mmap[:])
        # close mmap
        mmap._mmap.close()
        # augment by random horizontal flipping
        if self.augment:
            if random.random() < 0.5:
                rgb = np.fliplr(rgb).copy()
        # normalize into [-1, 1] range
        rgb = rgb / 127.5 - 1.0
        return rgb.transpose(2, 0, 1).astype(np.float32), label

    def __len__(self):
        ''' Total number of images in this database. '''
        return len(self.files)
