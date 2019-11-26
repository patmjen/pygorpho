"""
Thin wrappers to C bindings for gorpho. Only meant for internal use.
"""
import ctypes
import os
import numpy as np
import numpy.ctypeslib as ctl
import platform

class DummyFunc:
    pass

class DummyLib:
    def __getattr__(self, name):
        return DummyFunc()

# Load the shared library
def try_lib_load():
    """
    Try to load dynamic library with gorpho C bindings.

    Looks for the pygorpho shared library (eg. a .dll or .so) and loads it.
    The directory _thin.py is located in is searched as well as the parent
    directory. If PYGORPHO_PATH is set, this location is also searched.

    Returns
    -------
    (library, path)
        Tuple with library object return by numpy.ctypes.load_library and the
        path to the library.

    Raises
    ------
    ImportError
        If the dynamic library file could not be found.
    """
    # If we are building the documentation, then we abort the import
    rtd_build_environ = 'PYGORPHO_BUILD_READTHEDOCS'
    if rtd_build_environ in os.environ:
        import warnings
        warnings.warn('Environment variable {} exists - we assume documentation is being built and are aborting the import'.format(rtd_build_environ))
        return DummyLib(), __file__

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

DILATE = 0 # Must match MOP_DILATE in pygorpho.cuh
ERODE = 1 # Must match MOP_ERODE in pygorpho.cuh

def raise_on_error(error_code):
    """
    Raise appropriate exception if error code is not 0

    Parameters
    ----------
    error_code
        Error code returned by C bindings to gorpho. See pygorpho.cuh.

    Raises
    ------
    ValueError
        If error_code is 1-3 or an unknown value.
    RuntimeError
        If error_code is 4-5
    """
    if error_code == 0: # SUCCESS
        return
    elif error_code == 1: # ERR_BAD_MORPH_OP
        raise ValueError('invalid morhology operation code')
    elif error_code == 2: # ERR_BAD_TYPE
        raise ValueError('invalid type')
    elif error_code == 3: # ERR_BAD_CUDA_DEVICE
        raise ValueError('invalid device number')
    elif error_code == 4: # ERR_NO_AVAILABLE_CUDA_DEVICE
        return RuntimeError('no CUDA device available')
    elif error_code == 5: # ERR_UNCAUGHT_EXCEPTION
        return RuntimeError('an unaught C++ exception occured')
    else:
        raise ValueError('invalid error code: {}'.format(error_code))


get_device_count_impl = PYGORPHO_LIB.pyGetDeviceCount

get_device_name_impl = PYGORPHO_LIB.pyGetDeviceName
get_device_name_impl.argtypes = [
    ctypes.c_int,
    ctypes.c_char_p
]

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

flat_dilate_erode_impl = PYGORPHO_LIB.pyFlatDilateErode
flat_dilate_erode_impl.argtypes = [
    ctypes.c_void_p,                               # res
    ctypes.c_void_p,                               # vol
    ctl.ndpointer(dtype=np.dtype('?'), flags='C'), # strel
    ctypes.c_int,                                  # volX
    ctypes.c_int,                                  # volY
    ctypes.c_int,                                  # volZ
    ctypes.c_int,                                  # strelX
    ctypes.c_int,                                  # strelY
    ctypes.c_int,                                  # strelZ
    ctypes.c_int,                                  # type
    ctypes.c_int,                                  # op
    ctypes.c_int,                                  # blockX
    ctypes.c_int,                                  # blockY
    ctypes.c_int,                                  # blockZ
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