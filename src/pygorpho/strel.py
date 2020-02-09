"""Structuring elements for mathematical morhology"""
import ctypes
import numpy as np
from . import _thin
from . import constants

def flatBallApprox(radius, type=constants.BEST):
    """
    Returns approximation to flat ball of radius using line segments.

    The approximation is constructed according to [J19]_.

    Parameters
    ----------
    r
        Integer radius of flat ball.
    type
        Whether to constrain the zonohedral approximation inside or outside
        the sphere. Must either INSIDE, BEST, or OUTSIDE from constants.

    Returns
    -------
    (numpy.array, numpy.array)
        Tuple with step vectors and line lengths which parameterizes the line
        segments.

    References
    ----------
    .. [J19] P. M. Jensen et al., "Zonohedral Approximation of Spherical
       Structuring Element for Volumetric Morphology," Scandinavian
       Conference on Image Analysis (pp. 128-139). Springer. 2019.
    """
    LINE_COUNT = 13
    assert (type == constants.INSIDE or type == constants.BEST or
        type == constants.OUTSIDE)

    radius = ctypes.c_int(radius)
    type = ctypes.c_int(type)

    line_steps = np.empty((LINE_COUNT, 3), dtype=np.int32, order='C')
    line_lens = np.empty(LINE_COUNT, dtype=np.int32)

    ret = _thin.flat_ball_approx_impl(line_steps, line_lens, radius, type)
    _thin.raise_on_error(ret)

    return (line_steps, line_lens)