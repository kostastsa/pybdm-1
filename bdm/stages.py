"""Block decomposition stage functions."""
from collections import Counter
import numpy as np
from .encoding import string_from_array


def partition(x, shape, step=None):
    """General dataset partition function.

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
    if len(set(shape)) != 1:
        raise AttributeError(f"Partition shape has to be symmetric not {shape}")
    if len(shape) != x.ndim:
        x = x.squeeze()
    if len(shape) != x.ndim:
        raise AttributeError("Dataset and part shapes are not conformable")
    if not step:
        step = shape[0]
    shapes = list(zip(x.shape, shape))
    if all([k <= l for k, l in shapes ]):
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
                    yield from partition(x[idx], shape)
                break

def partition_ignore(x, shape):
    """Partition with ignore leftovers boundary condition.

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
    for part in partition(x, shape):
        if part.shape == shape:
            yield part

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
            if '-' in key:
                cmx = ctm[key]
            else:
                cmx = ctm.get(key, ctm[key.lstrip('0')])
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
