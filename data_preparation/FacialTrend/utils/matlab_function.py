def sub2ind(sz, I1, I2):
    """
    sub2ind: matlab function
    convert subscripts to linear indices

    param sz: size of array ([height, width])
    param I1, I2: multidimensional subscripts (numpy 1-d array,)

    return ind: linear indices
    """
    ind = []
    for i in range(I1.shape[0]):
        row = I1[i]
        col = I2[i]
        ind.append(int(row + (col - 1) * sz[0]) - 1)

    return ind


if __name__ == '__main__':
    import numpy as np
    print(sub2ind([3,5], np.array([1,2,3]), np.array([3,4,5])))