"""Mathematical morphology with general (grayscale) structuring elements."""

import numpy as np
from . import _thin
from . import constants

def dilate_erode(vol, strel, op, blockSize=[256,256,256]):
    """
    Dilation/erosion with general structuring element.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to numpy array of at most
        3 dimensions.
    strel
        Structuring element.  Must be convertible to numpy array of at most 3
        dimensions.
    op
        Operation to perform. Must be either DILATE or ERODE from constants.
    blockSize
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation/erosion.
    """
    assert (op == constants.DILATE or op == constants.ERODE)

    # Recast inputs to correct datatype
    old_shape = vol.shape
    vol = np.atleast_3d(np.asarray(vol))
    strel = np.atleast_3d(np.asarray(strel, dtype=vol.dtype))
    assert vol.dtype == strel.dtype

    # Prepare output volume
    volSize = vol.shape
    res = np.empty(volSize, dtype=vol.dtype)

    ret = _thin.gen_dilate_erode_impl(
        res.ctypes.data, vol.ctypes.data, strel.ctypes.data,
        volSize[2], volSize[1], volSize[0],
        strel.shape[2], strel.shape[1], strel.shape[0],
        vol.dtype.num, op,
        blockSize[2], blockSize[1], blockSize[0])
    _thin.raise_on_error(ret)

    return np.resize(res, old_shape)


def dilate(vol, strel, blockSize=[256,256,256]):
    """
    Dilation with general structuring element.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to numpy array of at most
        3 dimensions.
    strel
        Structuring element.  Must be convertible to numpy array of at most 3
        dimensions.
    blockSize
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation/erosion.
    """
    return dilate_erode(vol, strel, constants.DILATE, blockSize)


def erode(vol, strel, blockSize=[256,256,256]):
    """
    Erosion with general structuring element.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to numpy array of at most
        3 dimensions.
    strel
        Structuring element.  Must be convertible to numpy array of at most 3
        dimensions.
    blockSize
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation/erosion.
    """
    return dilate_erode(vol, strel, constants.ERODE, blockSize)