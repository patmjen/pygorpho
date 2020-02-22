"""Mathematical morphology with general (grayscale) structuring elements."""

import numpy as np
from . import _thin
from . import constants


def dilate_erode(vol, strel, op, block_size=[256, 256, 256]):
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
        Operation to perform. Must be either ``DILATE`` or ``ERODE`` from
        constants.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation/erosion.
    """
    assert (op == constants.DILATE or op == constants.ERODE)

    # Recast inputs to correct datatype
    vol = np.asarray(vol)
    old_shape = vol.shape
    vol = np.atleast_3d(vol)
    strel = np.atleast_3d(np.asarray(strel, dtype=vol.dtype))
    assert vol.dtype == strel.dtype

    # Prepare output volume
    vol_size = vol.shape
    res = np.empty(vol_size, dtype=vol.dtype)

    ret = _thin.gen_dilate_erode_impl(
        res.ctypes.data, vol.ctypes.data, strel.ctypes.data,
        vol_size[2], vol_size[1], vol_size[0],
        strel.shape[2], strel.shape[1], strel.shape[0],
        vol.dtype.num, op,
        block_size[2], block_size[1], block_size[0])
    _thin.raise_on_error(ret)

    return np.resize(res, old_shape)


def dilate(vol, strel, block_size=[256, 256, 256]):
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
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation/erosion.
    """
    return dilate_erode(vol, strel, constants.DILATE, block_size)


def erode(vol, strel, block_size=[256, 256, 256]):
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
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation/erosion.
    """
    return dilate_erode(vol, strel, constants.ERODE, block_size)
