
import glob

import cv2
import dlib
import numpy as np
from tqdm import tqdm

from Model3D.utils.inference import crop_img, parse_roi_box_from_landmark


def get_crop_img(img_list, face_regressor, face_detector):
    """
    crop the facial region from a image

    param img_list: the list of images to be cropped

    return img_crop: dictionary of cropped images, the key is the path to the image, the value is the cropped image
    """
    # print('===== Cropping images =====')
    img_crop = {}
    for img_path in tqdm(img_list):
        print(img_path)
        img_ori = cv2.imread(img_path)
        if img_ori is None:
            print('image error')
            continue

        rects = face_detector(img_ori, 1)
        if len(rects) == 0:
            # print('no face')
            continue

        for rect in rects:
            pts = face_regressor(img_ori, rect).parts()
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
            roi_box = parse_roi_box_from_landmark(pts)

            img_crop[img_path] = crop_img(img_ori, roi_box)

    return img_crop


def get_crop_img_single(img, face_regressor, face_detector):
    """
    crop the facial region from a image

    param img: the image to be cropped

    return img_crop: the cropped image
    """
    # print('===== Cropping images =====')
    if img is None:
        print('image error')
        return None

    rects = face_detector(img, 1)
    if len(rects) == 0:
        # print('no face')
        return None
    
    for rect in rects:
        pts = face_regressor(img, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T
        roi_box = parse_roi_box_from_landmark(pts)
        img_crop = crop_img(img, roi_box)

    return img_crop
