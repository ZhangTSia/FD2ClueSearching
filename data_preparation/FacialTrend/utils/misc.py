import numpy as np
from time import time


def paramap_pose(pose_para):
    """
    split parameters
    """

    pose_para = np.squeeze(pose_para).tolist()
    phi = pose_para[0]
    gamma = pose_para[1]
    theta = pose_para[2]
    t3dx = pose_para[3]
    t3dy = pose_para[4]
    t3dz = pose_para[5]
    f = pose_para[6]

    t3d = np.array([t3dx, t3dy, t3dz])

    return phi, gamma, theta, t3d, f


def rotation_matrix(phi, gamma, theta):
    """
    get the rotation matrix from rotation angle

    param phi: roll
    param gamma: pitch
    param theta: yaw

    return r: rotation matrix
    """

    r_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(phi), np.sin(phi)],
            [0, -np.sin(phi), np.cos(phi)]
        ]
    )
    r_y = np.array(
        [
            [np.cos(gamma), 0, -np.sin(gamma)],
            [0, 1, 0],
            [np.sin(gamma), 0, np.cos(gamma)],
        ]
    )
    r_z = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    r = np.dot(np.dot(r_x, r_y), r_z)

    return r


def Tnorm_VnormC(norm_tri, tri, n_tri, n_ver):
    """
    get norm of vertices from norm of triangles

    param normt: norm of each triangle (3*n_tri)
    param tri: triangle indices (3*n_tri)
    param ntri: number of triangles
    param nver: number of vertices

    return normv: norm of each vertex (3*n_ver)
    """
    norm_ver = np.zeros((n_ver, 3))

    start = time()
    for i in range(n_tri):
        pt0, pt1, pt2 = tri[i] - 1
        # norm_ver[pt] += np.tile(norm_tri[i], (3, 1))
        norm_ver[pt0] += norm_tri[i]
        norm_ver[pt1] += norm_tri[i]
        norm_ver[pt2] += norm_tri[i]
    end = time()
    # print('norm used: {:.2f} s'.format(end - start))

    return norm_ver.T


def norm_direction(vertex, tri):
    """
    get the norm of each vertex

    param vertex: coordinates of vertices (3*n_ver)
    param tri: indices of the three points of triangles (3*n_tri)

    return N: norm of each vertex (3*n_ver)
    """
    # norm of each triangle
    pt1 = vertex[:, tri[0] - 1]
    pt2 = vertex[:, tri[1] - 1]
    pt3 = vertex[:, tri[2] - 1]
    norm_tri = np.cross((pt1 - pt2).T, (pt1 - pt3).T)

    # norm of each vertex
    N = Tnorm_VnormC(norm_tri, tri.T, tri.shape[1], vertex.shape[1])
    
    # normalize to unit length
    mag = np.sum(np.multiply(N, N), axis=0)
    # deal with zero vector
    co = np.where(mag == 0)
    mag[co] = 1
    N[0, co] = np.ones((len(co)))
    N = N / np.sqrt(np.tile(mag, (3, 1)))

    return N


def normalization(data):
    data_range = np.max(data) - np.min(data)
    return (data - np.min(data)) / data_range
