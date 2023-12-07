import os.path as osp
import os
from pathlib import Path
import numpy as np

import torch
import torch.utils.data as data
import cv2
import imageio
import scipy.io as sio
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random

def img_loader(path, input_size=None):
    img = np.array(imageio.imread(path))
    img = img.astype(np.float32) / 255
    return img

class ToTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


class Data_pncc2offset(data.Dataset):
    def __init__(self, img_pathes, pncc_pathes, target_pathes, filelist_fps, transform=None, random_flip=True, **kargs):
        self.img_pathes = img_pathes
        self.pncc_pathes = pncc_pathes
        self.target_pathes = target_pathes
        self.filelist_fps = filelist_fps
        self.transform = transform
        self.filelist = []
        self.random_flip = random_flip
        for di in range(len(img_pathes)):
            lines = Path(filelist_fps[di]).read_text().strip().split('\n')
            for line in lines:
                self.filelist.append((line, di))

        self.img_loader = img_loader

    def target_loader(self, path):
        offset_map = np.array(imageio.imread(path).astype(np.float32) / 255)
        #offset_map = offset_map * (self.offset_max - self.offset_min) + self.offset_min
        return offset_map


    def __getitem__(self, index):
        filename, di = self.filelist[index]
        #print(filename)
        if self.random_flip == True:
            if random.random()<0.5:
                flip = True
            else:
                flip = False

        if flip:
            img = self.img_loader(osp.join(self.img_pathes[di], filename+'.jpg'))
            img = cv2.flip(img,1)
            filename = filename + '_flip'
        else:
            img = self.img_loader(osp.join(self.img_pathes[di], filename+'.jpg'))
        
        pncc = self.img_loader(osp.join(self.pncc_pathes[di], filename + '.jpg'))
        input = np.concatenate((img,pncc), axis=2)

        target = self.target_loader(osp.join(self.target_pathes[di], filename+'.jpg'))
        target = np.transpose(target, (2, 0, 1))

        if self.transform is not None:
            input = self.transform(input)
        return input, target

    def __len__(self):
        return len(self.filelist)



class Data_Pncc2WeightedOffset(data.Dataset):
    def __init__(self, img_pathes, pncc_pathes, target_pathes, weight_pathes, filelist_fps, transform=None, random_flip=True, **kargs):
        self.img_pathes = img_pathes
        self.pncc_pathes = pncc_pathes
        self.target_pathes = target_pathes
        self.weight_pathes = weight_pathes
        self.filelist_fps = filelist_fps
        self.transform = transform
        self.filelist = []
        self.random_flip = random_flip
        for di in range(len(img_pathes)):
            lines = Path(filelist_fps[di]).read_text().strip().split('\n')
            for line in lines:
                self.filelist.append((line, di))

        self.img_loader = img_loader

    def target_loader(self, path):
        offset_map = np.array(imageio.imread(path).astype(np.float32) / 255)
        #offset_map = offset_map * (self.offset_max - self.offset_min) + self.offset_min
        return offset_map


    def __getitem__(self, index):
        filename, di = self.filelist[index]
        #print(filename)
        if self.random_flip == True:
            if random.random()<0.5:
                flip = True
            else:
                flip = False

        if flip:
            img = self.img_loader(osp.join(self.img_pathes[di], filename+'.jpg'))
            img = cv2.flip(img,1)
            filename = filename + '_flip'
        else:
            img = self.img_loader(osp.join(self.img_pathes[di], filename+'.jpg'))
        
        pncc = self.img_loader(osp.join(self.pncc_pathes[di], filename + '.jpg'))
        input = np.concatenate((img,pncc), axis=2)

        target = self.target_loader(osp.join(self.target_pathes[di], filename+'.jpg'))
        target = np.transpose(target, (2, 0, 1))

        weight = self.target_loader(osp.join(self.weight_pathes[di], filename+'.jpg'))
        weight = np.transpose(weight, (2, 0, 1))

        if self.transform is not None:
            input = self.transform(input)
        return input, target, weight

    def __len__(self):
        return len(self.filelist)




class Data_PnccHeatmap2WeightedOffset(data.Dataset):
    def __init__(self, img_pathes, pncc_pathes, heatmap_pathes, target_pathes, weight_pathes, filelist_fps, target_scale = 1.0, transform=None, random_flip=True, **kargs):
        self.img_pathes = img_pathes
        self.pncc_pathes = pncc_pathes
        self.heatmap_pathes = heatmap_pathes
        self.target_pathes = target_pathes
        self.weight_pathes = weight_pathes
        self.filelist_fps = filelist_fps
        self.transform = transform
        self.filelist = []
        self.random_flip = random_flip
        for di in range(len(img_pathes)):
            lines = Path(filelist_fps[di]).read_text().strip().split('\n')
            for line in lines:
                self.filelist.append((line, di))

        self.img_loader = img_loader
        self.target_scale = target_scale

    def target_loader(self, path):
        offset_map = np.array(imageio.imread(path).astype(np.float32) / 255)
        #offset_map = offset_map * (self.offset_max - self.offset_min) + self.offset_min
        return offset_map


    def __getitem__(self, index):
        filename, di = self.filelist[index]
        #print(filename)
        if self.random_flip == True:
            if random.random()<0.5:
                flip = True
            else:
                flip = False

        if flip:
            img = self.img_loader(osp.join(self.img_pathes[di], filename+'.jpg'))
            img = cv2.flip(img,1)
            filename = filename + '_flip'
        else:
            img = self.img_loader(osp.join(self.img_pathes[di], filename+'.jpg'))
        
        pncc = self.img_loader(osp.join(self.pncc_pathes[di], filename + '.jpg'))
        heatmap = self.img_loader(osp.join(self.heatmap_pathes[di], filename + '.jpg'))
        heatmap = heatmap[:,:,np.newaxis]
        input = np.concatenate((img,pncc,heatmap), axis=2)

        target = self.target_loader(osp.join(self.target_pathes[di], filename+'.jpg'))
        target = np.transpose(target, (2, 0, 1))
        target = target * self.target_scale

        weight = self.target_loader(osp.join(self.weight_pathes[di], filename+'.jpg'))
        weight = np.transpose(weight, (2, 0, 1))

        if self.transform is not None:
            input = self.transform(input)
        return input, target, weight

    def __len__(self):
        return len(self.filelist)

class Data_pncc_test(data.Dataset):
    def __init__(self, img_pathes, pncc_pathes, filelist_fps, transform=None, **kargs):
        self.img_pathes = img_pathes
        self.pncc_pathes = pncc_pathes
        self.filelist_fps = filelist_fps
        self.transform = transform
        self.filelist = []
        for di in range(len(img_pathes)):
            lines = Path(filelist_fps[di]).read_text().strip().split('\n')
            for line in lines:
                self.filelist.append((line, di))

        self.img_loader = img_loader

    def __getitem__(self, index):
        filename, di = self.filelist[index]

        # end = time.time()
        img = self.img_loader(osp.join(self.img_pathes[di], filename + '.jpg'))
        pncc = self.img_loader(osp.join(self.pncc_pathes[di], filename + '.jpg'))
        input = np.concatenate((img, pncc), axis=2)

        if self.transform is not None:
            input = self.transform(input)
        return input, filename, di

    def __len__(self):
        return len(self.filelist)