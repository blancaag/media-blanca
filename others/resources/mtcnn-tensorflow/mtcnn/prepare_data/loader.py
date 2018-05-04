import sys

import cv2
import numpy as np

from . import minibatch
from ..train_models.MTCNN_config import config


class TestLoader:
    #imdb image_path(list)
    def __init__(self, images, batch_size=1, shuffle=False):
        self.images = images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(images)  #num of data
        #self.index = np.arange(self.size)

        self.cur = 0
        self.data = None
        self.label = None

        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            #shuffle test image
            np.random.shuffle(self.images)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    #realize __iter__() and next()--->iterator
    #return iter object
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        image = self.images[self.cur]
        # `image` is an RGB image. But the model has been trained with
        # BGR, so converting.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.data = image


class ImageLoader:
    def __init__(self,
                 imdb,
                 im_size,
                 batch_size=config.BATCH_SIZE,
                 shuffle=False):

        self.imdb = imdb
        self.batch_size = batch_size
        self.im_size = im_size
        self.shuffle = shuffle

        self.cur = 0
        self.size = len(imdb)
        self.index = np.arange(self.size)
        self.num_classes = 2

        self.batch = None
        self.data = None
        self.label = None

        self.label_names = ['label', 'bbox_target']
        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data, self.label
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = minibatch.get_minibatch(imdb, self.num_classes,
                                              self.im_size)
        self.data = data['data']
        self.label = [label[name] for name in self.label_names]
