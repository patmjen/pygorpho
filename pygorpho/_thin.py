"""
Thin wrappers for internal use
"""
import ctypes
import os
import numpy as np
import numpy.ctypeslib as ctl
import platform

# Load the shared library
def try_lib_load():
    path_candidates = []
    # If PYGORPHO_PATH was set we start looking there
    if os.getenv('PYGORPHO_PATH') is not None:
        path_candidates.append(os.path.abspath(os.getenv('PYGORPHO_PATH')))
    # Look in the dir. where this python file is placed
    path_candidates.append(os.path.dirname(__file__))
    # Look in dir. one level up from current file dir.
    path_candidates.append(os.path.dirname(path_candidates[-1]))
    # Start looking
    for path in path_candidates:
        try:
            if platform.system() == 'Windows':
                lib = ctl.load_library('pygorpho', path)
            else:
                lib = ctl.load_library('libpygorpho', path)
            # Load was successful, so return path and lib now
            return lib, path
        except OSError:
            # Lib was not here so move on...
            pass
    else:
        raise ImportError('could not find pygorpho dynamic library file (try setting PYGORPHO_PATH environment variable)')


PYGORPHO_LIB, PYGORPHO_PATH = try_lib_load()

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