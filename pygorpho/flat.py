"""
Mathematical morphology with flat (binary) structuring elements.
"""
import numpy as np
from . import _thin
from . import constants

def linear_dilate_erode(vol, lineSteps, lineLens, op, blockSize=[256,256,512]):
    assert (op == constants.DILATE or op == constants.ERODE)

    # Recast inputs to correct datatype
    old_shape = vol.shape
    vol = np.atleast_3d(np.asarray(vol))
    lineSteps = np.atleast_2d(np.asarray(lineSteps, dtype=np.int32, order='C'))
    lineLens = np.asarray(lineLens, dtype=np.int32)
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
    return linear_dilate_erode(vol, lineSteps, lineLens, constants.DILATE, blockSize)


def linear_erode(vol, lineSteps, lineLens, blockSize=[256,256,512]):
    return linear_dilate_erode(vol, lineSteps, lineLens, constants.ERODE, blockSize)