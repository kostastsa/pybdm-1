"""Block decomposition stage functions."""
import numpy as np


def partition_ignore_leftover(x, shape):
    """Split method with ignore leftover boundary condition.

    Parameters
    ----------
    x : (N, k) array_like
        Dataset.
    shape : tuple
        Shape of parts.

    Yields
    ------
    str
        String representation of a dataset part.
    """
    if len(shape) != x.ndim:
        raise AttributeError("Dataset and part shapes are not conformable")
    shapes = list(zip(x.shape, shape))
    if all([k == l for k, l in shapes ]):
        yield x
    else:
        for k, shp in enumerate(shapes):
            n, step = shp
            if n > step:
                for i in range(0, n, step):
                    idx = tuple([
                        slice(i, i + step) if j == k else slice(None)
                        for j in range(x.ndim)
                    ])
                    yield from partition_ignore_leftover(x[idx], shape)
                break

def _array2str(arr):
    arr = np.apply_along_axis(''.join, 0, arr)
    for _ in range(arr.ndim):
        arr = np.apply_along_axis('-'.join, 0, arr)
    return str(arr)

def lookup(parts, ctm):
    """Lookup CTM values for parts in a reference dataset.

    Parameters
    ----------
    parts : sequence
        Ordered sequence of dataset parts.
    ctm : dict
        Reference CTM dataset.

    Yields
    ------
    tuple
        2-tuple with string representatio nof a dataset part and its CTM value.
    """
    for part in parts:
        key = _array2str(part)
        try:
            cmx = ctm[key]
        except KeyError:
            raise KeyError(f"CTM dataset does not contain object '{key}'")
        yield key, cmx


def aggregate(ctms):
    """Combine CTM of parts into BDM value.

    Parameters
    ----------
    ctms : sequence
        Ordered 1D sequence of CTM values.

    Returns
    -------
    float
        BDM value.
    """
    pass
