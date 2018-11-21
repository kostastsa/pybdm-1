"""Utility functions."""
import numpy as np


def array_from_string(x, sep='-', cast_to=int):
    """Make array from string representation.

    Parameters
    ----------
    x : str
        Array-string.
    sep : str
        Sequence separator.
    cast_to : type or None
        Cast array to given type. No casting if ``None``.
    """
    arr = np.array([ x for x in map(list, x.split(sep)) ])
    if cast_to:
        arr = arr.astype(cast_to)
    return arr

def string_from_array(arr, sep='-'):
    """Dump an array to its string representation.

    Parameters
    ----------
    arr : (N, k) array_like
        *Numpy* array.
    sep : str
        Sequence separator.
    """
    x = np.apply_along_axis(''.join, arr.ndim - 1, arr.astype(str))
    x = sep.join(np.ravel(x))
    return x
