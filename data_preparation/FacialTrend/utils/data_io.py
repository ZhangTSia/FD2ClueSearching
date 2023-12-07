import glob
import os

import cv2
import numpy as np
from scipy.io import loadmat

from .misc import normalization
from uvmap.uvmap import load_uv_coords


def load_uv(data_folder):
    """
    load parameters from BFM_UV
    """
    UV = loadmat(os.path.join(data_folder, 'BFM_UV.mat'))['UV']
    UV_paf = UV.copy(order='C')

    std_size = 256;
    UV[:,0] = UV[:,0] * (std_size - 1) + 1
    UV[:,1] = UV[:,1] * (std_size - 1) + 1
    UV[:, 1] = std_size + 1 - UV[:, 1]
    h, _ = UV.shape
    UV = np.concatenate((UV, np.zeros((h, 1))), axis=1).T

    return UV, UV_paf


def load_basic_for_v2(data_folder):
    tri = loadmat(os.path.join(data_folder, 'Model_BFM_PCA.mat'))['tri']
    mu_tex = np.squeeze(loadmat(os.path.join(data_folder, 'Model_BFM_PCA.mat'))['mu_tex'])
    w_tex = loadmat(os.path.join(data_folder, 'Model_BFM_PCA.mat'))['w_tex'][:, :20]
    sigma_tex = np.squeeze(loadmat(os.path.join(data_folder, 'Model_BFM_PCA.mat'))['sigma_tex'][0:20])
    face_front_bin = np.squeeze(loadmat(os.path.join(data_folder, 'Modelplus_face_bin.mat'))['face_front_bin'].astype(bool))
    UV, UV_paf = load_uv(data_folder)
    facial_detail_distribute = loadmat(os.path.join(data_folder, 'facial_detail_distribute.mat'))

    tri_full = loadmat(os.path.join(data_folder, 'tri_full.mat'))['tri_full']

    return tri, tri_full, w_tex, mu_tex, sigma_tex, face_front_bin, UV, UV_paf, facial_detail_distribute


def load_bfm(data_folder):
    """
    load parameters from BFM model
    """
    model = loadmat(os.path.join(data_folder, 'Model_Data.mat'))
    tri = model['tri']
    mu_tex = model['mu_tex']
    tex = np.reshape(mu_tex, (3, -1), 'F') / 255.
    face_front_bin = np.squeeze(model['face_front_bin'].astype(bool))

    return tex, tri, face_front_bin


def load_img(img_path):
    """
    load image and parameters of the image
    """
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.

    mat_folder = 'samples/mat'
    mat_path = os.path.join(mat_folder, os.path.split(img_path)[1].replace('.jpg', '_3D.mat'))
    pose = loadmat(mat_path)['Pose_Para']
    shape = loadmat(mat_path)['Shape_Para']
    exp = loadmat(mat_path)['Exp_Para']

    return img, pose, shape, exp


def load_img_vertex(file_path):
    """
    load image and parameters of the image
    """
    img = cv2.cvtColor(loadmat(file_path)['img'], cv2.COLOR_BGR2RGB) / 255
    vertex = loadmat(file_path)['vertex']

    return img, vertex


def save_img(filename, facial_trend, facial_detail):
    save_folder = 'results'
    save_folder_trend = os.path.join(save_folder, 'trend')
    save_folder_detail = os.path.join(save_folder, 'detail')
    os.makedirs(save_folder_trend, exist_ok=True)
    os.makedirs(save_folder_detail, exist_ok=True)

    save_path_trend = os.path.join(save_folder_trend, (os.path.splitext(os.path.split(filename)[-1])[0] + '_trend.jpg'))
    print(save_path_trend)
    facial_trend_save = normalization(facial_trend[:, :, [2, 1, 0]]) * 255
    cv2.imwrite(save_path_trend, facial_trend_save)

    save_path_detail = os.path.join(save_folder_detail, (os.path.splitext(os.path.split(filename)[-1])[0] + '_detail.jpg'))
    print(save_path_detail)
    facial_detail_save = normalization(facial_detail[:, :, [2, 1, 0]]) * 255
    cv2.imwrite(save_path_detail, facial_detail_save)


if __name__ == '__main__':
    UV = load_uv('../data')
    print(UV)