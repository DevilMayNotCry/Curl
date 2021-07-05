# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
import os
import os.path
import sys
import util
import numpy
import torch
import random
import logging
import torch.utils.data
numpy.set_printoptions(threshold=sys.maxsize)
from collections import defaultdict
from torch.autograd import Variable


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dict, transform=None, normaliser=1, for_test=False):
        self.transform = transform
        self.data_dict = data_dict
        self.normaliser = normaliser
        self.for_test = for_test

    def get_image(self, file_name, seed=None):
        image_file = util.ImageProcessing.load_image(
            file_name, normaliser=self.normaliser)
        image_file = image_file.astype(numpy.uint8)

        if(image_file.shape[2] == 4):
            image_file = numpy.delete(image_file, 3, -1)

        # 对图像进行规整  regular the images to better shape
        m = int(image_file.shape[0] / 8)
        n = int(image_file.shape[1] / 8)

        image_file = image_file[:8 * m, :8 * n]

        if(seed is not None):
            random.seed(seed)
            image_file = self.transform(image_file)

        return image_file


    def __len__(self):
        return (len(self.data_dict.keys()))

    def __getitem__(self, idx):
        while True:

            if idx in self.data_dict:
                seed = random.randint(0, 100000)
                return {
                    'input': self.get_image(self.data_dict[idx]['input'], seed=seed), 
                    'expertC_gt': self.get_image(self.data_dict[idx]['expertC_gt'], seed=seed),
                    'name': self.data_dict[idx]['input'].split("/")[-1]
                } if(not self.for_test) else {
                    'input': self.get_image(self.data_dict[idx]['input'], seed=seed), 
                    'name': self.data_dict[idx]['input'].split("/")[-1]
                }



class Adobe5kDataLoader():

    def __init__(self, data_dirpath, img_ids_filepath=None, for_test=False):
        self.data_dirpath = data_dirpath
        self.img_ids_filepath = img_ids_filepath
        self.data_dict = defaultdict(dict)
        self.for_test = for_test

    def load_data(self):
        if(not self.for_test):
            logging.info("Loading Adobe5k dataset ...")

            with open(self.img_ids_filepath) as f:
                image_ids = f.readlines()
                image_ids_list = [x.rstrip() for x in image_ids]

            for cnt, name in enumerate(image_ids_list):
                self.data_dict[cnt] = {
                    "input": os.path.join(self.data_dirpath, 'input', name + '.jpg'),
                    "expertC_gt": os.path.join(self.data_dirpath, "expertC_gt", name + '.jpg')
                } 
            return self.data_dict
        else:
            image_ids_list = os.listdir(self.data_dirpath)
            for cnt, name in enumerate(image_ids_list):
                self.data_dict[cnt] = {
                    "input": os.path.join(self.data_dirpath, name)
                }
            return self.data_dict



import torchvision.transforms as transforms

def get_loader(input_dir, for_test=False, batch_size=1, shuffle=False, num_workers=4, normaliser=1, img_ids_filepath=None):
    data_loader = Adobe5kDataLoader(data_dirpath=input_dir, for_test=for_test, img_ids_filepath=img_ids_filepath)
    data_dict = data_loader.load_data()
    _dataset = Dataset(data_dict=data_dict,
                       transform=transforms.Compose([transforms.ToTensor()]), 
                       normaliser=normaliser,
                       for_test=for_test)
    return torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


