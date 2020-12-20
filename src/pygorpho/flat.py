"""Mathematical morphology with flat (binary) structuring elements."""

import numpy as np
from . import _thin
from . import constants


def morph(vol, strel, op, block_size=[256, 256, 256]):
    """
    Morphological operation with flat structuring element.

    Parameters
    ----------
    vol
        Volume to apply operation to. Must be convertible to numpy array of at
        most 3 dimensions.
    strel
        Structuring element. Must be convertible to numpy array of at most 3
        dimensions.
    op
        Operation to perform. Must be either ``DILATE``, ``ERODE``, ``OPEN``,
        ``CLOSE``, ``TOPHAT``, ``CLOSE`` from ``constants``.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of the operation.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple dilation with an 11 x 11 x 11 box structuring element
        >>> vol = np.zeros((100, 100, 100))
        >>> vol[50, 50, 50] = 1
        >>> strel = np.ones((11, 11, 11))
        >>> res = pg.flat.morph(vol, strel, pg.DILATE)
    """
    assert(op in [constants.DILATE, constants.ERODE, constants.OPEN,
                  constants.CLOSE, constants.TOPHAT, constants.BOTHAT])

    # Recast inputs to correct datatype
    vol = np.asarray(vol)
    old_shape = vol.shape
    vol = np.atleast_3d(vol)
    strel = np.atleast_3d(np.asarray(strel, dtype=np.bool_))

    # Prepare output volume
    vol_size = vol.shape
    res = np.empty_like(vol)

    ret = _thin.flat_morph_op_impl(
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

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple dilation with an 11 x 11 x 11 box structuring element
        >>> vol = np.zeros((100, 100, 100))
        >>> vol[50, 50, 50] = 1
        >>> strel = np.ones((11, 11, 11))
        >>> res = pg.flat.dilate(vol, strel)
    """
    return morph(vol, strel, constants.DILATE, block_size)


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

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple erosion with an 11 x 11 x 11 box structuring element
        >>> vol = np.ones((100, 100, 100))
        >>> vol[50, 50, 50] = 0
        >>> strel = np.ones((11, 11, 11))
        >>> res = pg.flat.erode(vol, strel)
    """
    return morph(vol, strel, constants.ERODE, block_size)


def open(vol, strel, block_size=[256, 256, 256]):
    """
    Opening with flat structuring element.

    Parameters
    ----------
    vol
        Volume to open. Must be convertible to numpy array of at most
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
        Volume of same size as vol with the result of opening.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple opening with an 11 x 11 x 11 box structuring element
        >>> vol = np.zeros((100, 100, 100))
        >>> vol[10:15,10:15,48:53] = 1  # Small box
        >>> vol[60:80,60:80,40:60] = 1  # Big box
        >>> strel = np.ones((11, 11, 11))
        >>> res = pg.flat.open(vol, strel)
    """
    return morph(vol, strel, constants.OPEN, block_size)


def close(vol, strel, block_size=[256, 256, 256]):
    """
    Closing with flat structuring element.

    Parameters
    ----------
    vol
        Volume to close. Must be convertible to numpy array of at most
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
        Volume of same size as vol with the result of closing.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple closing with an 11 x 11 x 11 box structuring element
        >>> vol = np.ones((100, 100, 100))
        >>> vol[10:15,10:15,48:53] = 0  # Small box
        >>> vol[60:80,60:80,40:60] = 0  # Big box
        >>> strel = np.ones((11, 11, 11))
        >>> res = pg.flat.close(vol, strel)
    """
    return morph(vol, strel, constants.CLOSE, block_size)


def tophat(vol, strel, block_size=[256, 256, 256]):
    """
    Top-hat transform with flat structuring element.

    Also known as a white top hat transform.
    It is given by ``tophat(x) = x - open(x)``.

    Parameters
    ----------
    vol
        Volume to top-hat transform. Must be convertible to numpy array of at
        most 3 dimensions.
    strel
        Structuring element. Must be convertible to numpy array of at most 3
        dimensions.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of the top-hat transform.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple top-hat with an 11 x 11 x 11 box structuring element
        >>> vol = np.zeros((100, 100, 100))
        >>> vol[10:15,10:15,48:53] = 1  # Small box
        >>> vol[60:80,60:80,40:60] = 1  # Big box
        >>> strel = np.ones((11, 11, 11))
        >>> res = pg.flat.tophat(vol, strel)
    """
    return morph(vol, strel, constants.TOPHAT, block_size)


def bothat(vol, strel, block_size=[256, 256, 256]):
    """
    Bot-hat transform with flat structuring element.

    Also known as a black top-hat transform.
    It is given by ``bothat(x) = close(x) - x``.

    Parameters
    ----------
    vol
        Volume to bot-hat transform. Must be convertible to numpy array of at
        most 3 dimensions.
    strel
        Structuring element. Must be convertible to numpy array of at most 3
        dimensions.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of the bot-hat transform.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple bot-hat with an 11 x 11 x 11 box structuring element
        >>> vol = np.ones((100, 100, 100))
        >>> vol[10:15,10:15,48:53] = 0  # Small box
        >>> vol[60:80,60:80,40:60] = 0  # Big box
        >>> strel = np.ones((11, 11, 11))
        >>> res = pg.flat.bothat(vol, strel)
    """
    return morph(vol, strel, constants.BOTHAT, block_size)


def linear_morph(vol, line_steps, line_lens, op, block_size=[256, 256, 512]):
    """
    Morphological operation with flat line segment structuring elements.

    Performs a morphological operation volume with a sequence of flat line
    segments. Line segments are parameterized with a (integer) step vector and
    a length giving the number of steps. The operation is the same for all line
    segments.

    The operations are performed using the van Herk/Gil-Werman algorithm
    [H92]_ [GW93]_.

    Parameters
    ----------
    vol
        Volume to apply operation to. Must be convertible to numpy array of at
        most 3 dimensions.
    line_steps
        Step vector or sequence of step vectors. A step vector must have
        integer coordinates and control the direction of the line segment.
    line_lens
        Length or sequence of lengths. Controls the length of the line
        segments. A length of 0 leaves the volume unchanged.
    op
        Operation to perform for all line segments. Must be either ``DILATE``
        or ``ERODE`` from ``constants``.
    block_size
        Block size for GPU processing. Volume is sent to the GPU in blocks of
        this size.

    Returns
    -------
    numpy.array
        Volume of same size as vol with the result of the operation.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple dilation with an 11 x 15 x 21 box structuring element
        >>> vol = np.zeros((100,100,100))
        >>> vol[50, 50, 50] = 1
        >>> lineSteps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> lineLens = [11, 15, 21]
        >>> res = pg.flat.linear_morph(vol, lineSteps, lineLens, pg.DILATE)

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
    res = np.empty_like(vol)

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
        Volume to dilate. Must be convertible to a numpy array of at most 3
        dimensions.
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

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple dilation with an 11 x 15 x 21 box structuring element
        >>> vol = np.zeros((100,100,100))
        >>> vol[50, 50, 50] = 1
        >>> lineSteps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> lineLens = [11, 15, 21]
        >>> res = pg.flat.linear_dilate(vol, lineSteps, lineLens)
    """
    return linear_morph(vol, line_steps, line_lens, constants.DILATE,
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
        Volume to erode. Must be convertible to a numpy array of at most 3
        dimensions.
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

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple erosion with an 11 x 15 x 21 box structuring element
        >>> vol = np.ones((100,100,100))
        >>> vol[50, 50, 50] = 0
        >>> lineSteps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> lineLens = [11, 15, 21]
        >>> res = pg.flat.linear_erode(vol, lineSteps, lineLens)
    """
    return linear_morph(vol, line_steps, line_lens, constants.ERODE,
                        block_size)


def linear_open(vol, line_steps, line_lens, block_size=[256, 256, 512]):
    """
    Opening with flat line segment structuring elements.

    Opens volume with a sequence of flat line segments. Line segments are
    parameterized with a (integer) step vector and a length giving the number
    of steps. The operations is the same for all line segments.

    The operations are performed using the van Herk/Gil-Werman algorithm
    [H92]_ [GW93]_.

    Parameters
    ----------
    vol
        Volume to open. Must be convertible to a numpy array of at most 3
        dimensions.
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
        Volume of same size as vol with the result of opening.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple opening with an 11 x 11 x 11 box structuring element
        >>> vol = np.zeros((100, 100, 100))
        >>> vol[10:15,10:15,48:53] = 1  # Small box
        >>> vol[60:80,60:80,40:60] = 1  # Big box
        >>> lineSteps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> lineLens = [11, 11, 11]
        >>> res = pg.flat.linear_open(vol, lineSteps, lineLens)
    """
    res = linear_erode(vol, line_steps, line_lens, block_size)
    return linear_dilate(res, line_steps, line_lens, block_size)


def linear_close(vol, line_steps, line_lens, block_size=[256, 256, 512]):
    """
    Closing with flat line segment structuring elements.

    Closes volume with a sequence of flat line segments. Line segments are
    parameterized with a (integer) step vector and a length giving the number
    of steps. The operations is the same for all line segments.

    The operations are performed using the van Herk/Gil-Werman algorithm
    [H92]_ [GW93]_.

    Parameters
    ----------
    vol
        Volume to close. Must be convertible to a numpy array of at most 3
        dimensions.
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
        Volume of same size as vol with the result of closing.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple closing with an 11 x 11 x 11 box structuring element
        >>> vol = np.ones((100, 100, 100))
        >>> vol[10:15,10:15,48:53] = 0  # Small box
        >>> vol[60:80,60:80,40:60] = 0  # Big box
        >>> lineSteps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> lineLens = [11, 11, 11]
        >>> res = pg.flat.linear_close(vol, lineSteps, lineLens)
    """
    res = linear_dilate(vol, line_steps, line_lens, block_size)
    return linear_erode(res, line_steps, line_lens, block_size)


def linear_tophat(vol, line_steps, line_lens, block_size=[256, 256, 512]):
    """
    Top-hat transform with flat line segment structuring elements.

    Top-hat transforms volume with a sequence of flat line segments. Line
    segments are parameterized with a (integer) step vector and a length giving
    the number of steps. The operations is the same for all line segments.

    The operations are performed using the van Herk/Gil-Werman algorithm
    [H92]_ [GW93]_.

    Parameters
    ----------
    vol
        Volume to top-hat transform. Must be convertible to a numpy array of at
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
        Volume of same size as vol with the result of top-hat transform.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple top-hat with an 11 x 11 x 11 box structuring element
        >>> vol = np.ones((100, 100, 100))
        >>> vol[10:15,10:15,48:53] = 0  # Small box
        >>> vol[60:80,60:80,40:60] = 0  # Big box
        >>> lineSteps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> lineLens = [11, 11, 11]
        >>> res = pg.flat.linear_tophat(vol, lineSteps, lineLens)
    """
    return vol - linear_open(vol, line_steps, line_lens, block_size)


def linear_bothat(vol, line_steps, line_lens, block_size=[256, 256, 512]):
    """
    Bot-hat transform with flat line segment structuring elements.

    Bot-hat transforms volume with a sequence of flat line segments. Line
    segments are parameterized with a (integer) step vector and a length giving
    the number of steps. The operations is the same for all line segments.

    The operations are performed using the van Herk/Gil-Werman algorithm
    [H92]_ [GW93]_.

    Parameters
    ----------
    vol
        Volume to bot-hat transform. Must be convertible to a numpy array of at
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
        Volume of same size as vol with the result of bot-hat transform.

    Example
    -------
    .. code-block:: python
        :dedent: 4

        >>> import numpy as np
        >>> import pygorpho as pg
        >>> # Simple bot-hat with an 11 x 11 x 11 box structuring element
        >>> vol = np.ones((100, 100, 100))
        >>> vol[10:15,10:15,48:53] = 0  # Small box
        >>> vol[60:80,60:80,40:60] = 0  # Big box
        >>> lineSteps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> lineLens = [11, 11, 11]
        >>> res = pg.flat.linear_tophat(vol, lineSteps, lineLens)
    """
    return linear_close(vol, line_steps, line_lens, block_size) - vol
