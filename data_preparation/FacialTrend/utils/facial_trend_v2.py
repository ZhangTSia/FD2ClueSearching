import os
from glob import glob

import numpy as np
from scipy.io import loadmat

from FacialTrend.utils.misc import norm_direction

from .data_io import load_basic_for_v2, load_uv
from .render import Mex_ZBuffer, crender_colors, render_colors

np.random.seed(0)


def facial_detail_distribute_aid(img, vertex, tri, mu_tex, w_tex, sigma_tex, norm, valid_bin, UV):
    # TODO unfinished here
    return 0
    # return facial_amb_dir_ctex_img, facial_amb_dir_ctex_uv , facial_itex_img, facial_amb_ctex_img, facial_amb_ctex_uv, facial_dir_itex_img , alpha_harmonic, alpha_harmonic_amb, alphas_tex, alphas_tex_amb


def facial_detail_distribute(data_folder, dataset_folder):
    # TODO unfinished here
    return 0
    # UV = load_uv(data_folder)
    # tri, mu_tex, w_tex, sigma_tex, face_front_bin = load_basic_for_v2(data_folder)
    
    # facial_itex_img_all = []
    # facial_dir_itex_img_all = []

    # file_list = glob(os.path.join(dataset_folder, '*.mat'))
    # for i in file_list:
    #     img = loadmat(i)['img'][:, :, [2, 1, 3]]
    #     project_vertex = loadmat(i)['vertex']
        
    #     norm = norm_direction(project_vertex, tri)
    #     visibility = norm[2] < 0

    #     facial_amb_dir_ctex_img, facial_amb_dir_ctex_uv , facial_itex_img, facial_amb_ctex_img, facial_amb_ctex_uv, facial_dir_itex_img , alpha_harmonic, alpha_harmonic_amb, alphas_tex, alphas_tex_amb = facial_detail_distribute_aid(project_Vertex, tri, mu_tex, w_tex, sigma_tex, norm, np.squeeze(face_front_bin & visibility), img, UV)
        
    #     facial_itex_pixels = []
    #     facial_dir_itex_pixels = []
    #     for c in range(3):
    #         temp = facial_itex_img[:,:,c]
    #         temp = temp[temp != 0].T
    #         rand_ind = np.random.randint(1, temp.shape[0], temp.shape[0])
    #         temp = temp[rand_ind[:1000]]
            
    #         facial_itex_pixels.append(temp)
            
    #         temp = facial_dir_itex_img[:, :, c]
    #         temp = temp[temp != 0].T
    #         rand_ind = np.random.randint(1, temp.shape[0], temp.shape[0])
    #         temp = temp[rand_ind[:1000]]
    #         facial_dir_itex_pixels.append(temp)
        
    #     facial_itex_img_all.append(facial_itex_pixels)
    #     facial_dir_itex_img_all.append(facial_dir_itex_pixels)

    # facial_itex_img_all = np.asarray(facial_itex_img_all)
    # facial_dir_itex_img_all = np.asarray(facial_dir_itex_img_all)

    # facial_itex_mean = np.mean(facial_itex_img_all(:));
    # facial_itex_std = np.std(facial_itex_img_all(:));
    # facial_itex_min = facial_itex_mean - 3 * facial_itex_std;
    # facial_itex_max = facial_itex_mean + 3 * facial_itex_std;

    # facial_dir_itex_mean = np.mean(facial_dir_itex_img_all(:));
    # facial_dir_itex_std = np.std(facial_dir_itex_img_all(:));
    # facial_dir_itex_min = facial_dir_itex_mean - 3 * facial_dir_itex_std;
    # facial_dir_itex_max = facial_dir_itex_mean + 3 * facial_dir_itex_std;

    # return facial_itex_mean, facial_itex_std, facial_itex_min, facial_itex_max, facial_dir_itex_mean, facial_dir_itex_std, facial_dir_itex_min, facial_dir_itex_max


def under_full_light(img, vertex, tri, mu_tex, mu_tex_valid, w_tex, w_tex_valid, sigma_tex, tex_pixel, valid_bin, UV, facial_detail_distribute, harmonic, _lambda, max_iterations):
    (height, width, n_channel) = img.shape

    facial_itex_min = facial_detail_distribute['facial_itex_min']
    facial_itex_max = facial_detail_distribute['facial_itex_max']

    Hs = []
    Ys = []
    Ys1 = []
    for i in range(n_channel):
        Hs.append(harmonic[valid_bin])
        Ys.append(tex_pixel[i, valid_bin].T)
        Ys1 = np.append(Ys1, tex_pixel[i, valid_bin].T)

    alphas = [0] * 3
    alpha_tex = np.zeros(w_tex_valid.shape[1])

    for i in range(max_iterations):
        common_tex = mu_tex_valid + w_tex_valid @ alpha_tex
        common_tex = np.reshape(common_tex, (3, -1), 'F')
        # 1. get harmonic coefficients
        for c in range(n_channel):
            # solve the Y_current = A * alpha
            H = Hs[c]
            Y = Ys[c]
            
            left = H * np.tile(common_tex[c], (harmonic.shape[1], 1)).T
            right = Y
            alpha, _, _, _ = np.linalg.lstsq(left, right, -1)

            alphas[c] = alpha

        H = []
        for c in range(n_channel):
            H = np.append(H, harmonic[valid_bin] @ alphas[c])
        
        left = w_tex_valid * np.tile(H, (w_tex_valid.shape[1], 1)).T
        right = Ys1 - mu_tex_valid * H
        right = left.T @ right
        left = left.T @ left + np.eye(left.shape[1]) @ np.diag(sigma_tex) * _lambda
        alpha_tex, _, _, _ = np.linalg.lstsq(left, right, -1)

    common_tex = mu_tex + w_tex @ alpha_tex
    common_tex = np.reshape(common_tex, (3, -1), 'F')
    common_tex = np.minimum(np.maximum(common_tex, 0), 1)

    facial_amb_dir_ctex = np.zeros(common_tex.shape)
    for c in range(n_channel):
        temp = harmonic @ alphas[c] * common_tex[c]
        facial_amb_dir_ctex[c] = temp.T
    
    facial_amb_dir_ctex = np.minimum(np.maximum(facial_amb_dir_ctex, 0), 1)
    facial_amb_dir_ctex_img, valid_map = crender_colors(vertex.T, tri.T, facial_amb_dir_ctex.T, img)

    facial_itex_img = img - facial_amb_dir_ctex_img

    valid_map = np.tile(valid_map[:, :, np.newaxis], (1, 1, 3))

    facial_itex_img = (facial_itex_img - facial_itex_min / (facial_itex_max - facial_itex_min)) * valid_map

    facial_amb_dir_ctex_uv, tri_ind = crender_colors(UV.T, tri.T, facial_amb_dir_ctex.T, np.zeros((256, 256, 3)))

    return facial_amb_dir_ctex_img, facial_amb_dir_ctex_uv, facial_itex_img, alphas, alpha_tex, valid_map


def under_amb_light(img, vertex, tri, mu_tex, mu_tex_valid, w_tex, w_tex_valid, sigma_tex, tex_pixel, valid_bin, UV, facial_detail_distribute, harmonic, _lambda, max_iterations, valid_map):
    (height, width, n_channel) = img.shape
    harmonic_dim = 1
    facial_dir_itex_min = facial_detail_distribute['facial_dir_itex_min']
    facial_dir_itex_max = facial_detail_distribute['facial_dir_itex_max']

    Hs = []
    Ys = []
    Ys1 = []
    for i in range(n_channel):
        Hs.append(harmonic[valid_bin, 0])
        Ys.append(tex_pixel[i, valid_bin].T)
        Ys1 = np.append(Ys1, tex_pixel[i, valid_bin].T)

    alphas_amb = [0] * 3
    alpha_tex_amb = np.zeros(w_tex_valid.shape[1])

    for i in range(max_iterations):
        common_tex = mu_tex_valid + w_tex_valid @ alpha_tex_amb
        common_tex = np.reshape(common_tex, (3, -1), 'F')
        # 1. get harmonic coefficients
        for c in range(n_channel):
            # solve the Y_current = A @ alpha
            H = Hs[c]
            Y = Ys[c]
            
            left = H * common_tex[c]
            right = Y
            alpha, _, _, _ = np.linalg.lstsq(left[:, np.newaxis], right, -1)
            
            alphas_amb[c] = alpha
            
        H = []
        for c in range(n_channel):
            H = np.append(H, harmonic[valid_bin, 0] * alphas_amb[c])

        left = w_tex_valid * np.tile(H, (w_tex_valid.shape[1], 1)).T
        right = Ys1 - mu_tex_valid * H
        right = left.T @ right
        left = left.T @ left + np.eye(left.shape[1]) @ np.diag(sigma_tex) * _lambda
        alpha_tex_amb, _, _, _ = np.linalg.lstsq(left, right, -1)

    common_tex_amb = mu_tex + w_tex @ alpha_tex_amb
    common_tex_amb = np.reshape(common_tex_amb, (3, -1), 'F')
    common_tex_amb = np.minimum(np.maximum(common_tex_amb, 0), 1)

    facial_amb_ctex = np.zeros(common_tex_amb.shape)
    for c in range(n_channel):
        temp = harmonic[:, 0] * alphas_amb[c] * common_tex_amb[c]
        facial_amb_ctex[c] = temp

    facial_amb_ctex = np.minimum(np.maximum(facial_amb_ctex, 0), 1)
    facial_amb_ctex_img, _ = crender_colors(vertex.T, tri.T, facial_amb_ctex.T, img)
    facial_dir_itex_img = img - facial_amb_ctex_img

    facial_dir_itex_img = (facial_dir_itex_img - facial_dir_itex_min) / (facial_dir_itex_max - facial_dir_itex_min) * valid_map

    facial_amb_ctex_uv, tri_ind = crender_colors(UV.T, tri.T, facial_amb_ctex.T, np.zeros((256,256,3)))

    return facial_amb_ctex_img, facial_amb_ctex_uv, facial_dir_itex_img, alphas_amb, alpha_tex_amb


if __name__ == '__main__':
    # facial_itex_mean, facial_itex_std, facial_itex_min, facial_itex_max, facial_dir_itex_mean, facial_dir_itex_std, facial_dir_itex_min, facial_dir_itex_max = facial_detail_distribute('..')
    # import cv2
    # img = cv2.imread('0000.png')
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # print(img)
    print(np.random.randint(1,10, 10))
