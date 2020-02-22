"""Mathematical morphology with flat (binary) structuring elements."""

import numpy as np
from . import _thin
from . import constants


def dilate_erode(vol, strel, op, block_size=[256, 256, 256]):
    """
    Dilation/erosion with flat structuring element.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to numpy array of at most
        3 dimensions.
    strel
        Structuring element. Must be convertible to numpy array of at most 3
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
    strel = np.atleast_3d(np.asarray(strel, dtype=np.bool_))

    # Prepare output volume
    vol_size = vol.shape
    res = np.empty(vol_size, dtype=vol.dtype)

    ret = _thin.flat_dilate_erode_impl(
        res.ctypes.data, vol.ctypes.data, strel,
        vol_size[2], vol_size[1], vol_size[0],
        strel.shape[2], strel.shape[1], strel.shape[0],
        vol.dtype.num, op,
        block_size[2], block_size[1], block_size[0])
    _thin.raise_on_error(ret)

    return np.resize(res, old_shape)


def dilate(vol, strel, block_size=[256, 256, 256]):
    """
    Dilation with flat structuring element.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to numpy array of at most
        3 dimensions.
    strel
        Structuring element. Must be convertible to numpy array of at most 3
        dimensions.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation.
    """
    return dilate_erode(vol, strel, constants.DILATE, block_size)


def erode(vol, strel, block_size=[256, 256, 256]):
    """
    Erosion with flat structuring element.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to numpy array of at most
        3 dimensions.
    strel
        Structuring element. Must be convertible to numpy array of at most 3
        dimensions.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of erosion.
    """
    return dilate_erode(vol, strel, constants.ERODE, block_size)


def linear_dilate_erode(vol, line_steps, line_lens, op,
                        block_size=[256, 256, 512]):
    """
    Dilation/erosion with flat line segment structuring elements.

    Dilates/erodes volume with a sequence of flat line segments. Line segments
    are parameterized with a (integer) step vector and a length giving the
    number of steps. The operation is the same for all line segments.

    The operations are performed using the van Herk/Gil-Werman algorithm
    [H92]_ [GW93]_.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to numpy array of at most
        3 dimensions.
    line_steps
        Step vector or sequence of step vectors. A step vector must have
        integer coordinates and control the direction of the line segment.
    line_lens
        Length or sequence of lengths. Controls the length of the line
        segments. A length of 0 leaves the volume unchanged.
    op
        Operation to perform for all line segments. Must be either ``DILATE``
        or ``ERODE`` from constants.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation/erosion.

    References
    ----------
    .. [H92] M. Van Herk, "A fast algorithm for local minimum and maximum
       filters on rectangular and octagonal kernels," Pattern Recognition
       Letters 13. (pp. 517-521). 1992.

    .. [GW93] J. Gil and M Werman, "Computing 2-D min, median, and max
       filters," IEEE Transactions on Pattern Analysis and Machine
       Intelligence 24. (pp. 504-507). 1993.
    """
    assert (op == constants.DILATE or op == constants.ERODE)

    # Recast inputs to correct datatype
    vol = np.asarray(vol)
    old_shape = vol.shape
    vol = np.atleast_3d(vol)
    line_steps = np.atleast_2d(
        np.asarray(line_steps, dtype=np.int32, order='C'))
    line_lens = np.atleast_1d(np.asarray(line_lens, dtype=np.int32))
    assert line_steps.ndim == 2
    assert line_steps.shape[1] == 3
    assert line_steps.shape[0] == line_lens.shape[0]

    line_steps = np.array(np.flip(line_steps, axis=1))

    # Prepare output volume
    vol_size = vol.shape
    res = np.empty(vol_size, dtype=vol.dtype)

    ret = _thin.flat_linear_dilate_erode_impl(
        res.ctypes.data, vol.ctypes.data, line_steps, line_lens,
        vol_size[2], vol_size[1], vol_size[0],
        line_lens.shape[0],
        vol.dtype.num, op,
        block_size[2], block_size[1], block_size[0])
    _thin.raise_on_error(ret)

    return np.resize(res, old_shape)


def linear_dilate(vol, line_steps, line_lens, block_size=[256, 256, 512]):
    """
    Dilation with flat line segment structuring elements.

    Erodes volume with a sequence of flat line segments. Line segments are
    parameterized with a (integer) step vector and a length giving the number
    of steps. The operations is the same for all line segments.

    The operations are performed using the van Herk/Gil-Werman algorithm
    [H92]_ [GW93]_.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to a numpy array of at
        most 3 dimensions.
    line_steps
        Step vector or sequence of step vectors. A step vector must have
        integer coordinates and control the direction of the line segment.
    line_lens
        Length or sequence of lengths. Controls the length of the line
        segments. A length of 0 leaves the volume unchanged.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of dilation.
    """
    return linear_dilate_erode(vol, line_steps, line_lens, constants.DILATE,
                               block_size)


def linear_erode(vol, line_steps, line_lens, block_size=[256, 256, 512]):
    """
    Erosion with flat line segment structuring elements.

    Erodes volume with a sequence of flat line segments. Line segments are
    parameterized with a (integer) step vector and a length giving the number
    of steps. The operations is the same for all line segments.

    The operations are performed using the van Herk/Gil-Werman algorithm
    [H92]_ [GW93]_.

    Parameters
    ----------
    vol
        Volume to dilate/erode. Must be convertible to a numpy array of at
        most 3 dimensions.
    line_steps
        Step vector or sequence of step vectors. A step vector must have
        integer coordinates and control the direction of the line segment.
    line_lens
        Length or sequence of lengths. Controls the length of the line
        segments. A length of 0 leaves the volume unchanged.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of erosion.
    """
    return linear_dilate_erode(vol, line_steps, line_lens, constants.ERODE,
                               block_size)
