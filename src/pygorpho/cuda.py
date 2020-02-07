"""Query functions to get information about available CUDA devices"""
import ctypes
from . import _thin

def get_device_count():
    """
    Returns the number of available CUDA devices.

    Returns
    -------
    int
        Number of available CUDA devices.
    """
    return _thin.get_device_count_impl()


def get_device_name(device):
    """
    Returns the name of queried CUDA device.

    Parameters
    ----------
    device
        ID of CUDA device.

    Returns
    -------
    str
        Name of queried CUDA device.
    """
    buffer = ctypes.create_string_buffer(256)
    ret = _thin.get_device_name_impl(device, buffer)
    _thin.raise_on_error(ret)
    return buffer.value.decode() # Decode to return a proper string