"""Mathematical morphology with flat (binary) structuring elements."""

import numpy as np
from . import _thin
from . import constants

def linear_dilate_erode(vol, lineSteps, lineLens, op, blockSize=[256,256,512]):
    """
    Dilation/erosion with flat line segment structuring elements.

    Dilates/erodes volume with a sequence of flat line segments. Line segments
    are parameterized with a (integer) step vector and a length giving the
    number of steps. The operations is the same for all line segments.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to numpy array of at most
        3 dimensions.
    lineSteps
        Step vector or sequence of step vectors. A step vector must have
        integer coordinates and control the direction of the line segment.
    lineLens
        Length or sequence of lengths. Controls the length of the line
        segments. A length of 0 leaves the volume unchanged.
    op
        Operation to perform for all line segments. Must be either DILATE or
        ERODE from constants.
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
    lineSteps = np.atleast_2d(np.asarray(lineSteps, dtype=np.int32, order='C'))
    lineLens = np.atleast_1d(np.asarray(lineLens, dtype=np.int32))
    assert lineSteps.ndim == 2
    assert lineSteps.shape[0] == lineLens.shape[0]

    # Prepare output volume
    volSize = vol.shape
    res = np.empty(volSize, dtype=vol.dtype)

    ret = _thin.flat_linear_dilate_erode_impl(
        res.ctypes.data, vol.ctypes.data, lineSteps, lineLens,
        volSize[2], volSize[1], volSize[0],
        lineLens.shape[0],
        vol.dtype.num, op,
        blockSize[2], blockSize[1], blockSize[0])
    _thin.raise_on_error(ret)

    return np.resize(res, old_shape)


def linear_dilate(vol, lineSteps, lineLens, blockSize=[256,256,512]):
    """
    Dilation with flat line segment structuring elements.

    Erodes volume with a sequence of flat line segments. Line segments are
    parameterized with a (integer) step vector and a length giving the number
    of steps. The operations is the same for all line segments.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to a numpy array of at
        most 3 dimensions.
    lineSteps
        Step vector or sequence of step vectors. A step vector must have
        integer coordinates and control the direction of the line segment.
    lineLens
        Length or sequence of lengths. Controls the length of the line
        segments. A length of 0 leaves the volume unchanged.
    blockSize
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation.
    """
    return linear_dilate_erode(vol, lineSteps, lineLens, constants.DILATE, blockSize)


def linear_erode(vol, lineSteps, lineLens, blockSize=[256,256,512]):
    """
    Erosion with flat line segment structuring elements.

    Erodes volume with a sequence of flat line segments. Line segments are
    parameterized with a (integer) step vector and a length giving the number
    of steps. The operations is the same for all line segments.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to a numpy array of at
        most 3 dimensions.
    lineSteps
        Step vector or sequence of step vectors. A step vector must have
        integer coordinates and control the direction of the line segment.
    lineLens
        Length or sequence of lengths. Controls the length of the line
        segments. A length of 0 leaves the volume unchanged.
    blockSize
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of erosion.
    """
    return linear_dilate_erode(vol, lineSteps, lineLens, constants.ERODE, blockSize)