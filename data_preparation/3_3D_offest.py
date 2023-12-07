import os, sys

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
# sys.path.append('E:\\PAMI\\pytorch_3D_offset')

import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging
import scipy.io as sio
import imageio
import skimage

import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from networks.network import PRNet as net
from datasets.datasets import Data_pncc2offset, ToTensor
from Model3D.utils.render import crender
from Model3D.utils.inference import crop_img, crop_img_inv, parse_roi_box_from_landmark
from Model3D.utils.heatmap import points_to_heatmap

from Model3D.utils.params import *
import matplotlib.pyplot as plt

# global args (configuration)
args = None
lr = None
STD_SIZE = 256

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

test_3ddfa = True


def eval_main():
    mean_shape = sio.loadmat('Model3D/models/mean_shape.mat')['vertex']
    tri_full = sio.loadmat('Model3D/models/tri_full.mat')['tri_full']
    keypoints68 = np.squeeze(sio.loadmat('Model3D/models/keypoints68.mat')['keypoints'])
    keypoints106 = np.squeeze(sio.loadmat('Model3D/models/keypoints106.mat')['keypoints106'])
    numVertex = mean_shape.shape[1]
    snapshot_name = 'Model3D/models/_checkpoint_epoch_50.pth.tar'

    root_path = '../data'
    videos_path = os.path.join(root_path, 'videos')
    video_list = os.listdir(videos_path)
    transform = transforms.Compose([ToTensor()])
    ## load_model
    model = net(in_channels=3 + 3 + 1, out_channels=3)
    model = nn.DataParallel(model).cuda()
    if os.path.isfile(snapshot_name):
        print("=> loading checkpoint '{}'".format(snapshot_name))
        # checkpoint = torch.load(resume_name, map_location='cuda:0')
        checkpoint = torch.load(snapshot_name, map_location=lambda storage, loc: storage)['state_dict']
        model.load_state_dict(checkpoint)
    else:
        print("no checkpoint at {resume_name}")
        return

    model.eval()

    for video_name in tqdm(video_list):
        img_path = os.path.join(root_path, 'face', video_name.replace('.mp4', '')) + '/'
        init_path = os.path.join(root_path, '3DDFA', video_name.replace('.mp4', '')) + '/'
        save_path = os.path.join(root_path, 'offset', video_name.replace('.mp4', '')) + '/'
        evaluate(img_path, init_path, save_path, model, tri_full, keypoints68, keypoints106, numVertex, transform)


def evaluate(img_path, init_path, save_path, model, tri_full, keypoints68, keypoints106, numVertex, transform,
             target_scale=10.0):
    count = 0
    # lines = open(filelist_file).read().strip().split('\n')
    filelist = os.listdir(img_path)
    print(img_path)
    for filename in filelist:
        # print(filename)
        filename = filename.replace('.png', '')
        if not (os.path.exists((img_path + filename + '.png'))):  # jpg
            print(img_path + filename + '.png')
            count = count + 1
            print('facelost{}/{}'.format(count, len(filelist)))
            continue

        if not (os.path.exists((init_path + filename + '_3D.mat'))):
            count = count + 1
            print('3Dlost{}/{}'.format(count, len(filelist)))
            continue

        img_ori = cv2.imread(img_path + filename + '.png').astype(np.float32) / 255.
        height, width, nChannels = img_ori.shape

        # Get reconstructed vertex by 3DDFA
        temp = sio.loadmat(init_path + filename + '_3D.mat')
        R = temp['R']
        t3d = temp['t3d']
        alpha_shp = temp['alpha_shape']
        alpha_exp = temp['alpha_exp']
        shape_dt = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
        vertex_dt = R @ shape_dt + np.tile(t3d, (1, numVertex))
        # vertex_dt[1,:] = -vertex_dt[1,:] + height
        # crop the image
        vertex_bbox = vertex_dt[:, keypoints68].copy()
        vertex_bbox[1, :] = -vertex_bbox[1, :] + height
        roi_box = parse_roi_box_from_landmark(vertex_bbox)
        img_crop = crop_img(img_ori, roi_box)

        height_crop, width_crop, nChannels = img_crop.shape
        crop_scale = STD_SIZE / height_crop
        img_crop = cv2.resize(img_crop, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)

        subpath = filename
        ind = subpath.rfind('/')
        subpath = subpath[:ind + 1]

        # pncc
        vertex_dt[0, :] = vertex_dt[0, :] - roi_box[0]
        vertex_dt[1, :] = vertex_dt[1, :] + roi_box[1] + height_crop - height
        vertex_dt = vertex_dt * crop_scale
        vertex_dt[1, :] = STD_SIZE - vertex_dt[1, :]
        pncc_feature = crender(img_crop, vertex_dt, pncc_code, tri_full)

        # plt.imshow(img_crop + 0.7 * pncc_feature)
        # plt.show()

        pt2d = vertex_dt[0:2, keypoints106]
        heatmap = points_to_heatmap(pt2d, (STD_SIZE, STD_SIZE), 2)
        input = np.concatenate((img_crop, pncc_feature, heatmap), axis=2)
        input = transform(input).unsqueeze(0)
        output = model(input)
        output = output.squeeze().cpu().detach().numpy().astype(np.float32)
        output = output / target_scale
        output = np.transpose(output, (1, 2, 0))
        output = output / crop_scale

        output = cv2.resize(output, dsize=(width_crop, height_crop), interpolation=cv2.INTER_LINEAR)

        output = crop_img_inv(output, roi_box, height, width)

        subpath = filename + '.mat'
        ind = subpath.rfind('/')
        subpath = subpath[:ind + 1]

        temp = save_path + subpath
        if not os.path.exists(temp):
            os.makedirs(temp)

        # sio.savemat(save_path + filename + '.mat', {'output': output})
        # print(save_path + filename + '.jpg')
        cv2.imwrite(save_path + filename + '.jpg', output)

        count = count + 1
        print('{}/{}'.format(count, len(filelist)),end='')


if __name__ == '__main__':
    eval_main()
