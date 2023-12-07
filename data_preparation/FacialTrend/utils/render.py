import numpy as np

from .cython import mesh_core_cython


"""
Reference:
    https://github.com/YadiraF/PRNet/blob/master/utils/render.py
    https://github.com/cleardusk/3DDFA/blob/master/utils/render.py
"""


def is_point_in_tri(point, tri_points):
    """
    judge if a point is in a triangle
    Method: http://blackpawn.com/texts/pointinpoly/

    param point: the point to be judged ([u, v] or [x, y])
    param tri_points: three vertices(2d points) of a triangle (3*[x, y])

    return bool: true if in triangle
    """
    tp = tri_points

    # vectors
    v0 = tp[:, 2] - tp[:, 0]
    v1 = tp[:, 1] - tp[:, 0]
    v2 = point - tp[:, 0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00 * dot11 - dot01 * dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)


def render_colors(vertex, tri, tex, img_src):
    """
    render mesh by z buffer

    param vertex: coordinates of vertices (3*n_ver)
    param tri: indices of the three points of triangles (3*n_tri)
    param tex: texture (3*n_ver)
    param img_src: RGB image (img_height*img_width*3)

    return img: facial trend img
    """
    # initial
    height, width, n_channel = img_src.shape
    img = np.zeros((height, width, n_channel))

    depth_buffer = np.zeros([height, width]) - 999999.0
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertex[2, tri[0] - 1] + vertex[2, tri[1] - 1] + vertex[2, tri[2] - 1]) / 3.0
    tri_tex = (tex[:, tri[0] - 1] + tex[:, tri[1] - 1] + tex[:, tri[2] - 1]) / 3.0

    for i in range(tri.shape[1]):
        tri_idx = tri[:, i] - 1  # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertex[0, tri_idx]))), 0)
        umax = min(int(np.floor(np.max(vertex[0, tri_idx]))), width - 1)

        vmin = max(int(np.ceil(np.min(vertex[1, tri_idx]))), 0)
        vmax = min(int(np.floor(np.max(vertex[1, tri_idx]))), height - 1)

        if umax < umin or vmax < vmin:
            continue

        for u in range(umin, umax + 1):
            for v in range(vmin, vmax + 1):
                if tri_depth[i] > depth_buffer[v, u] and is_point_in_tri([u, v], vertex[:2, tri_idx]):
                    depth_buffer[v, u] = tri_depth[i]
                    img[v, u, :] = tri_tex[:, i]

    for i in range(height):
        for j in range(width):
            if np.all(img[i, j]) == 0:
                img[i, j] = img_src[i, j]

    return img


def crender_colors(vertex, tri, tex, img_src, BG=None):
    """
    render mesh with colors

    param vertex: coordinates of vertices (n_ver*3)
    param tri: indices of the three points of triangles (n_tri*3)
    param tex: texture (n_ver*3)
    param img_src: RGB image (img_height*img_width*3)
    param BG: background image

    return img: [height, width, n_channel]. rendered image./rendering.
    """
    height, width, n_channel = img_src.shape
    if BG is None:
        img = np.zeros((height, width, n_channel), dtype=np.float32)
    else:
        assert (
            BG.shape[0] == height and BG.shape[1] == width and BG.shape[2] == n_channel
        )
        img = BG.astype(np.float32).copy(order="C")
    depth_buffer = np.zeros([height, width], dtype=np.float32, order="C") - 999999.0

    # to C order
    vertex = vertex.astype(np.float32).copy(order="C")
    tri = tri.astype(np.int32).copy(order="C") - 1
    tex = tex.astype(np.float32).copy(order="C")

    mesh_core_cython.render_colors_core(
        img,
        vertex,
        tri,
        tex,
        depth_buffer,
        vertex.shape[0],
        tri.shape[0],
        height,
        width,
        n_channel,
    )

    tri_ind = np.ones((height, width))
    for i in range(height):
        for j in range(width):
            if np.all(img[i, j]) == 0:
                img[i, j] = img_src[i, j]
                tri_ind[i, j] = 0

    return img, tri_ind


def tri_cover_point(point, pt1, pt2, pt3):
    """
    judge if this triangle cover the point
    Method: http://blackpawn.com/texts/pointinpoly/

    param point: the point to judge ([u, v] or [x, y])
    param tri_point
    """
    v0 = pt3 - pt1  # C-A
    v1 = pt2 - pt1  # B-A
    v2 = point - pt1

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00 * dot11 - dot01 * dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)


def z_buffer(s2d, vertex, tri, tex, img_src):
    """
    z_buffer (needed to be fixed)

    param s2d: x-y coordinates of the vertices projected from 3D to 2D (2*n_ver)
    param vertex: coordinates of vertices (3*n_ver)
    param tri: indices of the three points of triangles (3*n_tri)
    param tex: texture (3*n_ver)
    param img_src: RGB image (img_height*img_width*3)

    return img: facial trend img
    """
    img = img_src.copy()
    height, width, n_channel = img_src.shape
    s2d[1] = height - s2d[1] + 1

    # for every triangles, find the point it covers
    imgr = np.zeros((height, width))

    point1 = s2d[:, tri[0] - 1]  # A point
    point2 = s2d[:, tri[1] - 1]  # B point
    point3 = s2d[:, tri[2] - 1]  # C point

    cent3d = (vertex[:, tri[0] - 1] + vertex[:, tri[1] - 1] + vertex[:, tri[2] - 1]) / 3.0
    r = np.sum(np.power(cent3d, 2), axis=0)

    tri_tex = (tex[:, tri[0] - 1] + tex[:, tri[1] - 1] + tex[:, tri[2] - 1]) / 3.0

    for i in range(tri.shape[1]):
        pt1 = point1[:, i]
        pt2 = point2[:, i]
        pt3 = point3[:, i]

        umin = int(np.ceil(np.min([pt1[0], pt2[0], pt3[0]])))
        umax = int(np.floor(np.max([pt1[0], pt2[0], pt3[0]])))

        vmin = int(np.ceil(np.min([pt1[1], pt2[1], pt3[1]])))
        vmax = int(np.floor(np.max([pt1[1], pt2[1], pt3[1]])))

        if umax < umin or vmax < vmin or umax > width or umin < 1 or vmax > height or vmin < 1:
            continue
        
        for u in range(umin, umax):
            for v in range(vmin, vmax):
                if imgr[v, u] < r[i] and tri_cover_point([u, v], pt1, pt2, pt3):
                    imgr[v, u] = r[i]
                    img[v, u] = tri_tex[:, i]

    return img


def Mex_ZBuffer(projectedVertex, tri, texture, img_src):
    pass