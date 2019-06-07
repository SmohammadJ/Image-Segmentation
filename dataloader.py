import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image

from torchvision import transforms

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]


    def applyDataAugmentation(self, data_image, label_image, i):

            # H FLIP
            if i == 1:
                data_image = transforms.functional.hflip(data_image)
                label_image = transforms.functional.hflip(label_image)

            # V FLIP
            if i == 2:
                data_image = transforms.functional.vflip(data_image)
                label_image = transforms.functional.vflip(label_image)

            # GAMMA CORRECTION
            if i == 3:
                gamma_value = random.uniform(1, 1.2)
                data_image = transforms.functional.adjust_gamma(data_image, gamma_value, gain=1)

            # HUE ADJUSTMENT
            if i == 4:
                hue_value = random.uniform(-0.1, 0.1)
                data_image = transforms.functional.adjust_hue(data_image, hue_value)

            # ZOOMING
            if i == 5:
                size = random.randint(600,800)
                data_image = np.asarray(transforms.functional.five_crop(data_image, size)[4])
                label_image = np.asarray(transforms.functional.five_crop(label_image, size)[4])
                data_image = Image.fromarray(data_image)
                label_image = Image.fromarray(label_image)
        
            return data_image, label_image



    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId :

            rand = random.randint(1,10)
            print("rand :    ",rand)


            data_image = Image.open(self.data_files[current])
            label_image = Image.open(self.label_files[current])

           

            if self.mode == 'train':
                data_image , label_image = self.applyDataAugmentation(data_image, label_image, rand)
            
            data_image = data_image.resize((388,388))
            label_image = label_image.resize((388,388))

            data_image = np.array(data_image)
            label_image = np.array(label_image)

            data_image = np.pad(data_image, (94,94), 'symmetric')

            data_image = data_image / 255

            current += 1
            yield (data_image, label_image)

            


      

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))