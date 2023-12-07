from math import pi

import numpy as np

from .matlab_function import sub2ind


def spherical_harmonic_basis(vertex, norm):
    """
    Spherical Harmonic Basis
    """

    harmonic_dim = 9
    nx = norm[0].T
    ny = norm[1].T
    nz = norm[2].T
    harmonic = np.zeros((vertex.shape[1], harmonic_dim))

    harmonic[:, 0] = np.sqrt(1 / (4 * pi)) * np.ones(vertex.shape[1])
    harmonic[:, 1] = np.sqrt(3 / (4 * pi)) * nx
    harmonic[:, 2] = np.sqrt(3 / (4 * pi)) * ny
    harmonic[:, 3] = np.sqrt(3 / (4 * pi)) * nz
    harmonic[:, 4] = 1 / 2 * np.sqrt(3 / (4 * pi)) * (2 * np.power(nz, 2) - np.power(nx, 2) - np.power(ny, 2))
    harmonic[:, 5] = 3 * np.sqrt(5 / (12 * pi)) * np.multiply(ny, nz)
    harmonic[:, 6] = 3 * np.sqrt(5 / (12 * pi)) * np.multiply(nx, nz)
    harmonic[:, 7] = 3 * np.sqrt(5 / (12 * pi)) * np.multiply(nx, ny)
    harmonic[:, 8] = 3 / 2 * np.sqrt(5 / (12 * pi)) * (np.power(nx, 2) - np.power(ny, 2))

    return harmonic


def facial_trend_fit(vertex, tex, norm, valid_bin, img, rend_lambda):
    """
    get the facial trend

    param vertex: coordinates of vertices (3*n_ver)
    param tex: texture (3*n_ver)
    param norm: norm of each vertex (3*n_ver)
    param valid_bin: visibility (1*53215)
    param img: RGB image (img_height*img_width*3)
    param rend_lambda:

    return appearance: facial trend texture (img_height*img_width*3)
    """

    max_iterations = 2

    height, width, n_channel = img.shape

    pt2d = vertex[[0, 1]]
    pt2d[0] = np.minimum(np.maximum(pt2d[0], 1), width)
    pt2d[1] = np.minimum(np.maximum(pt2d[1], 1), height)
    pt2d = np.round(pt2d)
    ind = sub2ind([height, width], pt2d[1], pt2d[0])

    tex_pixel = np.zeros(tex.shape)
    for i in range(n_channel):
        tex_pixel[i] = np.ravel(img[:, :, i], 'F')[ind]

    harmonic = spherical_harmonic_basis(vertex, norm)

    A_temp = []
    Y_temp = []
    for i in range(n_channel):
        temp = np.multiply(harmonic, np.tile(tex[i, np.newaxis].T, (1, harmonic.shape[1])))
        A_temp.append(temp[valid_bin])
        Y_temp.append(tex_pixel[i, valid_bin].T)
    A = np.concatenate((A_temp[0], A_temp[1], A_temp[2]))
    Y = np.concatenate((Y_temp[0], Y_temp[1], Y_temp[2]))

    light = np.ones(3)
    # get light
    for i in range(n_channel):
        XX = tex[i, valid_bin].T
        YY = tex_pixel[i, valid_bin].T
        light[i] = np.dot(XX.T, YY) / np.dot(XX.T, XX)

    n_ver_valid = np.flatnonzero(valid_bin).shape[0]
    regular_matrix = rend_lambda * np.identity(harmonic.shape[1])

    for i in range(max_iterations):
        # get harmonic coefficients
        Y_c = Y.copy()
        for j in range(n_channel):
            Y_c[j * n_ver_valid : j * n_ver_valid + n_ver_valid] = Y_c[j * n_ver_valid : j * n_ver_valid + n_ver_valid] / light[j]

        # solve the Y_current = A * alpha
        left = np.dot(A.T, Y_c)
        right = np.dot(A.T, A) + regular_matrix
        alpha = np.linalg.solve(right, left)

        # get light coefficients
        for j in range(n_channel):
            Y_c = Y[j * n_ver_valid : j * n_ver_valid + n_ver_valid]
            A_c = np.dot(A[j * n_ver_valid : j * n_ver_valid + n_ver_valid], alpha)
            light[j] = np.dot(A_c.T, Y_c) / np.dot(A_c.T, A_c)

    appearance = np.zeros(tex.shape)
    for i in range(n_channel):
        temp = np.dot(np.dot(harmonic * np.tile(tex[i, np.newaxis].T, (1, harmonic.shape[1])), alpha), light[i])
        appearance[i] = temp.T

    appearance = np.minimum(np.maximum(appearance, 0), 1)

    return appearance
