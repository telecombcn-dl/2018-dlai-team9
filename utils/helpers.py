import numpy as np
from PIL import Image
from skimage.color import lab2rgb


def unflatten_2d_array(pts_flt, pts_nd, axis=2):
    """
    Unflatten a 2d array with a certain axis
    :param pts_flt: prod(N \ N_axis) x M array
    :param pts_nd: N0xN1x...xNd array
    :param axis: Axis
    :type axis: int
    :return: N0xN1x...xNd array
    """
    shape = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, pts_nd.ndim), np.array(axis))
    axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
    axorder_rev = np.argsort(axorder)
    m = pts_flt.shape[1]
    new_shp = shape[nax].tolist()
    new_shp.append(m)
    pts_out = pts_flt.reshape(new_shp)
    pts_out = pts_out.transpose(axorder_rev)
    return pts_out


def flatten_nd_array(pts_nd, axis=2):
    """
    Flatten an nd array into a 2d array with a certain axis
    :param pts_nd: N0xN1x...xNd array
    :param axis: integer
    :return: prod(N \ N_axis) x N_axis array
    """
    shape = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, pts_nd.ndim), np.array(axis))
    axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
    pts_flt = pts_nd.transpose(axorder)
    pts_flt = pts_flt.reshape(np.prod(shape[nax]), shape[axis])
    return pts_flt


def show_image(img_array, encoding='RGB_norm'):
    if encoding == 'LAB':
        Image.fromarray(np.uint8(lab2rgb(img_array) * 255)).show()
    elif encoding == 'RGB_norm':
        Image.fromarray(np.uint8(img_array * 255)).show()
    elif encoding == 'RGB':
        Image.fromarray(img_array).show()
