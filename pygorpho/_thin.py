"""
Thin wrappers for internal use
"""
import ctypes
import os
import numpy as np
import numpy.ctypeslib as ctl

# Load the shared library
PYGORPHO_PATH = os.getenv('PYGORPHO_PATH','./')
PYGORPHO_LIB = ctl.load_library('pygorpho', PYGORPHO_PATH)

DILATE = PYGORPHO_LIB.pyDilateOp()
ERODE = PYGORPHO_LIB.pyErodeOp()

def raise_on_error(error_code):
    if error_code == 0: # SUCCESS
        return
    elif error_code == 1: # ERR_BAD_MORPH_OP
        raise ValueError('invalid morhology operation code')
    elif error_code == 2: # ERR_BAD_TYPE
        raise ValueError('invalid type')
    elif error_code == 3: # ERR_UNCAUGHT_EXCEPTION
        return RuntimeError('an unaught C++ exception occured')
    else:
        raise ValueError('invalid error code: {}'.format(error_code))


flat_ball_approx_impl = PYGORPHO_LIB.pyFlatBallApproxStrel
flat_ball_approx_impl.argtypes = [
    ctl.ndpointer(dtype=np.int32, flags='C'), # lineSteps 
    ctl.ndpointer(dtype=np.int32, flags='C'), # lineLens 
    ctypes.c_int,                             # radius
]

gen_dilate_erode_impl = PYGORPHO_LIB.pyGenDilateErode
gen_dilate_erode_impl.argtypes = [
    ctypes.c_void_p, # res
    ctypes.c_void_p, # vol
    ctypes.c_void_p, # strel
    ctypes.c_int,    # volX
    ctypes.c_int,    # volY
    ctypes.c_int,    # volZ
    ctypes.c_int,    # strelX
    ctypes.c_int,    # strelY
    ctypes.c_int,    # strelZ
    ctypes.c_int,    # type
    ctypes.c_int,    # op
    ctypes.c_int,    # blockX
    ctypes.c_int,    # blockY
    ctypes.c_int,    # blockZ
]

flat_linear_dilate_erode_impl = PYGORPHO_LIB.pyFlatLinearDilateErode
flat_linear_dilate_erode_impl.argtypes = [
    ctypes.c_void_p,                          # res
    ctypes.c_void_p,                          # vol
    ctl.ndpointer(dtype=np.int32, flags='C'), # lineSteps 
    ctl.ndpointer(dtype=np.int32, flags='C'), # lineLens 
    ctypes.c_int,                             # volX
    ctypes.c_int,                             # volY
    ctypes.c_int,                             # volZ
    ctypes.c_int,                             # numLines
    ctypes.c_int,                             # type
    ctypes.c_int,                             # op
    ctypes.c_int,                             # blockX
    ctypes.c_int,                             # blockY
    ctypes.c_int,                             # blockZ
]