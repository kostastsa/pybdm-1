"""Block decomposition stage functions."""
from collections import Counter
import numpy as np
from .encoding import string_from_array


def partition(x, shape, shift=0):
    """General dataset partition function.

    Parameters
    ----------
    x : (N, k) array_like
        Dataset.
    shape : tuple
        Shape of parts.
    shift : int
        Shift of the sliding window.
        Shift by partition shape if not positive.

    Yields
    ------
    array_like
        Dataset parts.
    """
    if len(set(shape)) != 1:
        raise AttributeError(f"Partition shape is not symmetric {shape}")
    if len(shape) != x.ndim:
        x = x.squeeze()
    if len(shape) != x.ndim:
        raise AttributeError("Dataset and part shapes are not conformable")
    shapes = list(zip(x.shape, shape))
    if all([k <= l for k, l in shapes ]):
        yield x
    else:
        for k, shp in enumerate(shapes):
            n, step = shp
            _shift = step if shift <= 0 else shift
            if n > step:
                end = n - step + 1 if shift > 0 else n
                for i in range(0, end, _shift):
                    idx = tuple([
                        slice(i, i + step) if j == k else slice(None)
                        for j in range(x.ndim)
                    ])
                    yield from partition(x[idx], shape, shift=shift)
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
    for part in partition(x, shape, shift=0):
        if part.shape == shape:
            yield part

def partition_shrink(x, shape, min_width=2):
    """Partition with shrinking shape boundary condition.

    Parameters
    ----------
    x (N, k) array_like
        Dataset.
    shape : tuple
        Shape of parts.
    min_width : int
        Minimal width of parts' shape.

    Yields
    ------
    array_like
        Dataset parts.
    """
    for part in partition(x, shape, shift=0):
        if part.shape == shape:
            yield part
        else:
            part_min_width = min(part.shape)
            _shape = tuple([ part_min_width for _ in range(len(shape)) ])
            if part_min_width >= min_width:
                yield from partition_shrink(part, _shape, min_width=min_width)

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
