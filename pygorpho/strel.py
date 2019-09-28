import ctypes
import numpy as np
from . import _thin

def flatBallApprox(radius):
    LINE_COUNT = 13
    radius = ctypes.c_int(radius)

    line_steps = np.empty((LINE_COUNT, 3), dtype=np.int32, order='C')
    line_lens = np.empty(LINE_COUNT, dtype=np.int32)
    
    ret = _thin.flat_ball_approx_impl(line_steps, line_lens, radius)
    _thin.raise_on_error(ret)

    return (line_steps, line_lens)