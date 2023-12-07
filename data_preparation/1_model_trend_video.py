import argparse
import glob
import os
import random
import subprocess
from os.path import join
from time import time

import cv2
import dlib
import numpy as np
import scipy.io as sio
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from scipy.io import savemat
from tqdm import tqdm

import Model3D.utils.mobilenet_v1 as mobilenet_v1
from extract_face import get_crop_img_single
from FacialTrend.utils.data_io import load_bfm, load_basic_for_v2, load_img_vertex
from Model3D.utils.ddfa import NormalizeGjz, ToTensorGjz
from Model3D.utils.estimate_pose import parse_pose
from Model3D.utils.inference import crop_img, parse_roi_box_from_landmark, predict_68pts, predict_dense
from uvmap.paf import gen_img_paf_from_vertex
from uvmap.uvmap import load_uv_coords


def img2detail(image):
    """
    load an image, return the trend and detail of that image

    param image: the image(bgr)

    return facial_trend, facial_detail
    """
    tex = np.reshape(mu_tex, (3, -1), 'F') / 255.
    img_crop = get_crop_img_single(image, face_regressor, face_detector)
    if img_crop is None:
        return None, None, None, None, None

    STD_SIZE = 120

    rects = face_detector(img_crop, 1)
    if len(rects) == 0:
        return None, None, None, None, None

    rect = rects[0]
    pts = face_regressor(img_crop, rect).parts()
    pts = np.array([[pt.x, pt.y] for pt in pts]).T
    roi_box = parse_roi_box_from_landmark(pts)

    img = crop_img(img_crop, roi_box)
    # print('===== 3DDFA =====')
    img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
    input = transform(img).unsqueeze(0)
    with torch.no_grad():
        if cuda:
            input = input.cuda()
        param = model(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

    # 68 pts
    pts68 = predict_68pts(param, roi_box)
    # dense face 3d vertices
    project_vertex = predict_dense(param, roi_box)  # projected vertices

    # ==============================
    # facial trend fit
    # ==============================
    facial_trend = None
    facial_detail = None
    appearance = None

    return facial_trend, facial_detail, img_crop, project_vertex, appearance


def extract_frames(video_path, video, data_savedir_dict):
    """
    Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent.
    """
    video_name = os.path.splitext(video)[0]
    for i in data_savedir_dict.keys():
        os.makedirs(os.path.join(data_savedir_dict[i], video_name), exist_ok=True)

    reader = cv2.VideoCapture(os.path.join(video_path, video))
    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success or frame_num >= args.num_frames:
            break

        img_name = '{:04d}.png'.format(frame_num)
        if os.path.exists(os.path.join(data_savedir_dict['face'], video_name, 'face_' + video_name + '_' + img_name)) and os.path.exists(os.path.join(data_savedir_dict['vertex'], video_name,'vertex_' + video_name + '_' + img_name.replace('.png', '.mat'))) and os.path.exists(os.path.join(data_savedir_dict['paf'], video_name, 'paf_' + video_name + '_' + img_name)):
            continue

        facial_trend, facial_detail, img_crop, project_vertex, appearance = img2detail(image)

        if img_crop is None:
            continue

        # savemat(os.path.join(data_savedir_dict['face'], video_name, 'face_' + video_name + '_' + img_name.replace('.png', '.mat')), {'face': img_crop[:, :, [2, 1, 0]]})
        cv2.imwrite(os.path.join(data_savedir_dict['face'], video_name, 'face_' + video_name + '_' + img_name),
                    img_crop)

        savemat(os.path.join(data_savedir_dict['vertex'], video_name,
                             'vertex_' + video_name + '_' + img_name.replace('.png', '.mat')),
                {'vertex': project_vertex})
        paf_feature = gen_img_paf_from_vertex(img_crop, project_vertex, tri_full, UV_paf, kernel_size=1)
        cv2.imwrite(os.path.join(data_savedir_dict['paf'], video_name, 'paf_' + video_name + '_' + img_name),
                    paf_feature)

        frame_num += 1
    reader.release()


def extract_method_videos():
    """
    Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure
    """
    videos_path = os.path.join(args.data_path, 'videos')
    print(videos_path)

    data_savedir_dict = {
        'face': os.path.join(args.data_path, 'face'),
        'vertex': os.path.join(args.data_path, 'vertex'),
        'paf': os.path.join(args.data_path, 'paf'),
    }

    for i in data_savedir_dict.keys():
        os.makedirs(data_savedir_dict[i], exist_ok=True)


    video_list = []
    if args.num_videos != -1:
        idx = random.sample(range(0, len(os.listdir(videos_path))), args.num_videos)
        file_list = os.listdir(videos_path)
        for i in idx:
            video_list.append(file_list[i])
    else:
        video_list = os.listdir(videos_path)

    for video in tqdm(video_list):
        extract_frames(videos_path, video, data_savedir_dict)


def main():
    # ==============================
    # process
    # ==============================
    extract_method_videos()


def test_single_img():
    # test on single image
    img_path = './test_img.png'
    image = cv2.imread(img_path)

    facial_trend, facial_detail, img_crop, project_vertex, appearance = img2detail(image)

    cv2.imwrite('./face_test_img.png', img_crop)

    savemat('./vertex_test_img.mat', {'vertex': project_vertex})
    paf_feature = gen_img_paf_from_vertex(img_crop, project_vertex, tri_full, UV_paf, kernel_size=1)

    # savemat(os.path.join(data_savedir_dict['paf'], video_name, 'paf_' + video_name + '_' + img_name.replace('.png', '.mat')), {'paf_feature': paf_feature[:, :, [2, 1, 0]]})
    cv2.imwrite('./paf_test_img.png', paf_feature)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # ==============================
    # args
    # ==============================
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_path', '-dp', default='../data', type=str)
    p.add_argument('--num_videos', '-nv', type=int, default=-1)
    p.add_argument('--num_frames', '-nf', type=int, default=100)
    args = p.parse_args()

    # ==============================
    # basic configuration
    # ==============================
    cuda = None  # torch.cuda.is_available()
    print("cuda:{}".format(cuda))

    data_folder = 'FacialTrend/data'
    tri, tri_full, w_tex, mu_tex, sigma_tex, face_front_bin, UV, UV_paf, facial_detail_distribute = load_basic_for_v2(
        data_folder)

    # ==============================
    # prepare model
    # ==============================
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
    if cuda:
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    dlib_landmark_model = 'Model3D/models/shape_predictor_68_face_landmarks.dat'
    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    face_detector = dlib.get_frontal_face_detector()

    # 3. load transform
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    
    main()
    # test_single_img()
