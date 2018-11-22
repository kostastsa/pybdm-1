"""Block decomposition stage functions."""
from collections import Counter
import numpy as np
from .encoding import string_from_array


def partition_ignore(x, shape):
    """Split method with ignore leftover boundary condition.

    Parameters
    ----------
    x : (N, k) array_like
        Dataset.
    shape : tuple
        Shape of parts.

    Yields
    ------
    array_like
        Dataset parts.
    """
    if len(shape) != x.ndim:
        if all([ d == 1 for d in x.shape ]):
            axes = tuple(range(len(shape), x.ndim))
            x = x.squeeze(axes)
        else:
            x = x.squeeze()
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
                    yield from partition_ignore(x[idx], shape)
                break

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
        2-tuple with string representation of a dataset part and its CTM value.
    """
    for part in parts:
        key = string_from_array(part)
        try:
            cmx = ctm[key]
        except KeyError:
            raise KeyError(f"CTM dataset does not contain object '{key}'")
        yield key, cmx


def aggregate(ctms):
    """Combine CTM of parts into BDM value.

    Parameters
    ----------
    ctms : sequence of 2-tuples
        Ordered 1D sequence of string keys and CTM values.

    Returns
    -------
    float
        BDM value.
    """
    counter = Counter()
    for key, ctm in ctms:
        counter.update([ (key, ctm) ])
    bdm = 0
    for key, n in counter.items():
        _, ctm = key
        bdm += ctm + np.log2(n)
    return bdm
