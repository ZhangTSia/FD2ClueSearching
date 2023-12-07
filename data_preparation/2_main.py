#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

import os.path
import random

from tqdm import tqdm

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.
"""

import torch
import torchvision.transforms as transforms
from Model3D.utils import mobilenet_v1
import numpy as np
import cv2
import dlib
from Model3D.utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from Model3D.utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, predict_3DMM_paras, dump_paras
from Model3D.utils.utils import *
import argparse
import torch.backends.cudnn as cudnn

STD_SIZE = 120


def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'Model3D/models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'Model3D/models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('Model3D/models/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    count = 0
    for img_fp in args.files:
        suffix = get_suffix(img_fp)
        count = count + 1
        if (os.path.exists(args.des_path + '{}_3D.mat'.format(img_fp.replace(suffix, '')))):
            continue

        img_ori = cv2.imread(args.path + img_fp)
        if not os.path.exists(args.bbox_path + img_fp[:-4] + '_boxpts.txt'):
            # continue
            rects = face_detector(img_ori, 1)
            # print(rects)
            rect = rects[0]
            # - use landmark for cropping
            pts = face_regressor(img_ori, rect).parts()
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
            roi_box = parse_roi_box_from_landmark(pts)
        else:
            pts_bbox = loadpts(args.bbox_path + img_fp[:-4] + '_boxpts.txt')
            roi_box = parse_roi_box_from_landmark(pts_bbox)

        img = crop_img(img_ori, roi_box)
        if img is None:
            print("*****")
            continue

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            if args.mode == 'gpu':
                input = input.cuda()
            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # 68 pts
        pts68 = predict_68pts(param, roi_box)
        R, t3d, alpha_shape, alpha_exp = predict_3DMM_paras(param, roi_box, img_ori.shape)

        # two-step for more accurate bbox to crop face
        if args.bbox_init == 'two':
            roi_box = parse_roi_box_from_landmark(pts68)
            img_step2 = crop_img(img_ori, roi_box)
            if img_step2 is None:
                continue

            img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img_step2).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = predict_68pts(param, roi_box)
            R, t3d, alpha_shape, alpha_exp = predict_3DMM_paras(param, roi_box, img_ori.shape)

        dump_paras(R, t3d, alpha_shape, alpha_exp, args.des_path + '{}_3D.mat'.format(img_fp.replace(suffix, '')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='true', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='false', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='false', type=str2bool)
    parser.add_argument('--dump_pts', default='false', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='false', type=str2bool)
    parser.add_argument('--dump_depth', default='false', type=str2bool)
    parser.add_argument('--dump_pncc', default='false', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--dump_paras', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='false', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')

    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--bbox_path', type=str, default='')
    parser.add_argument('--des_path', type=str, default='')

    args = parser.parse_args()

    args.bbox_init = 'two'
    args.dump_paras = 'true'
    # args.dump_paf = 'true'
    # args.dump_res = 'true'
    # args.show_flg = 'true'
    root_path = '../data'
    args.bbox_path = '../'
    DES_PATH = os.path.join(root_path, '3DDFA')
    if(not os.path.exists(DES_PATH)):os.mkdir(DES_PATH)
    videos_path = os.path.join(root_path, 'videos')
    video_list = os.listdir(videos_path)
    for video_name in tqdm(video_list):
        args.path = os.path.join(root_path, 'face', video_name.replace('.mp4', ''))
        args.des_path = os.path.join(DES_PATH, video_name.replace('.mp4', ''))
        if(not os.path.exists(args.des_path)):
            os.mkdir(args.des_path)
        files = get_all_files(args.path, 'png')
        args.files = files
        main(args)
