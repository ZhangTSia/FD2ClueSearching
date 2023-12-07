"""
input: ground truth 3d model, uvmap(U')
output: dist map

ground truth 3d model --> ground truth uvmap(U)
dist map = U - U'
"""

import os
from glob import glob
from tqdm import tqdm
from paf import gen_img_paf_from_vertex
import uvmap as uvmodule
from scipy.io import loadmat
import cv2
# import pdb; pdb.set_trace()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

uv_coords = uvmodule.load_uv_coords('BFM_UV.mat')
# shape_dt = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
# vertex_dt = R @ shape_dt + np.tile(t3d, (1,numVertex))

tri_full = loadmat('MM3D/model/tri_full.mat')['tri_full']

save_folder = 'test_result'
file_list = ['data/ffpp/0000.mat']
for fi in tqdm(file_list):
    file_path = fi
    img_crop = loadmat(file_path)['img']
    height, width, n_channel = img_crop.shape
    vertex = loadmat(file_path)['vertex']
    vertex_dt = vertex.copy()
    vertex_dt[1] = height + 1 - vertex_dt[1]

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(vertex_dt[0], vertex_dt[1], vertex_dt[2])
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # plt.show()

    # STD_SIZE = 120
    # img_crop = cv2.resize(img_crop, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
    paf_feature = gen_img_paf_from_vertex(img_crop, vertex_dt, tri_full, uv_coords, kernel_size=1)
    # print(paf_feature.shape)
    # uv_position_map = uvmodule.vertex2uvmap(vertex_dt, tri_full, uv_coords)
    # print(uv_position_map.shape)

    cv2.imwrite(os.path.join(save_folder, (fi.replace('/','_')).replace('.mat', '.jpg')), img_crop)
    cv2.imwrite(os.path.join(save_folder, (fi.replace('/','_')).replace('.mat', '_paf_feature.jpg')), paf_feature)
