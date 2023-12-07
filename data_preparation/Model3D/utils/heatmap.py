import numpy as np

def points_to_heatmap(points, heatmap_size, sigma):
    # resize points label

    # heatmap generation
    heatmap = np.zeros((heatmap_size[0], heatmap_size[1], 1))

    (dim, numpts) = points.shape

    for i in range(numpts):
        pt = points[:,i]
        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
        br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
        if (ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] or
                br[0] < 0 or br[1] < 0):
            # If not, just continue
            continue

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]
        # Image range
        heatmap_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
        heatmap_y = max(0, ul[1]), min(br[1], heatmap.shape[0])

        for j in range(heatmap_x[1] - heatmap_x[0]):
            for k in range(heatmap_y[1] - heatmap_y[0]):
                heatmap[heatmap_y[0] + k, heatmap_x[0] + j] = max(g[g_y[0] + k, g_x[0] + j],
                                                                    heatmap[heatmap_y[0] + k, heatmap_x[0] + j])

    return heatmap



def points_to_landmark_map(points, heatmap_num, heatmap_size, label_size, sigma):
    # resize points label
    for i in range(points.shape[0]):
        points[i] *= (float(heatmap_size[0]) / float(label_size[0]))

    # heatmap generation
    heatmap_lst = []

    for i in range(len(points)):
        heatmap = np.zeros((heatmap_size[0], heatmap_size[1], 1))

        pt = points[i]
        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
        br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
        if (ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] or
                br[0] < 0 or br[1] < 0):
            # If not, just continue
            continue

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]
        # Image range
        heatmap_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
        heatmap_y = max(0, ul[1]), min(br[1], heatmap.shape[0])

        if heatmap_num == 1:
            for j in range(heatmap_x[1] - heatmap_x[0]):
                for k in range(heatmap_y[1] - heatmap_y[0]):
                    heatmap[heatmap_y[0] + k, heatmap_x[0] + j] = max(g[g_y[0] + k, g_x[0] + j],
                                                                      heatmap[heatmap_y[0] + k, heatmap_x[0] + j])
        else:
            heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1], i] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        heatmap_lst.append(heatmap)

    heatmaps = np.concatenate(heatmap_lst, 2)


    return heatmaps
