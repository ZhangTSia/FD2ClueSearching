import os
import numpy as np


def get_all_files(dir, suffix):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path, 'png'))
        if os.path.isfile(path):
            if (path.endswith(suffix)):
                path = path[len(dir):]
                files_.append(path)
    return files_


def loadpts(filename):
    lines = open(filename).read().strip().split('\n')
    numpts = int(lines[0])
    lines = lines[1:]
    pts = np.ndarray((2, numpts))
    n = 0
    for l in lines:
        x, y = [float(_) for _ in l.split(' ')]
        pts[0, n] = x
        pts[1, n] = y
        n = n + 1

    return pts
