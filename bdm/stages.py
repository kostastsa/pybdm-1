"""Block decomposition stage functions.

Stage functions contain the main logic of the *Block Decomposition Method*
and its different flavours depending on boundary conditions etc.
The implementation follows the *split-apply-combine* approach.
In the first stage an input dataset is partitioned into parts of shape
appropriate for the selected CTM dataset (split stage).
Next, approximate complexity for parts based on the *Coding Theorem Method*
is looked up in the reference (apply stage) dataset.
Finally, values for individual parts are aggregated into a final BDM value
(combine stage).

The partition (split) stage is implemented by the family of ``partition_*`` functions.
The lookup (apply) stage is implemented by the family of ``lookup_*`` functions.
The aggregate (combine) stage is implented by the family of ``aggregate_*`` functions.

The general principle is that specific functions should in most cases
be wrappers around the core family functions, which accordingly are:

* :py:func:`bdm.stages.partition`
* :py:func:`bdm.stages.lookup`
* :py:func:`bdm.stages.aggregate`
"""
from collections import Counter
import numpy as np
from .encoding import string_from_array


def partition(x, shape, shift=0):
    """Core partition stage function.

    It is implemented as a generator that yields subsequent parts.

    Parameters
    ----------
    x : array_like
        Dataset of arbitrary dimensionality represented by a *Numpy* array.
    shape : tuple
        Shape of parts.
    shift : int
        Shift of the sliding window.
        In general, if positive, should not be greater than ``1``.
        Shift by partition shape if not positive.

    Yields
    ------
    array_like
        Dataset parts.

    Raises
    ------
    AttributeError
        If parts' `shape` is equal in each dimension.
        If parts' `shape` and dataset's shape are not conformable.

    Examples
    --------
    >>> [ x for x in partition(np.ones((3, 3), dtype=int), shape=(2, 2)) ]
    [array([[1, 1],
           [1, 1]]), array([[1],
           [1]]), array([[1, 1]]), array([[1]])]
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

    In this variant parts that can not be further sliced and fitted into
    the desired `shape` are simply omitted.

    Parameters
    ----------
    x : array_like
        Dataset of arbitrary dimensionality represented by a *Numpy* array.
    shape : tuple
        Shape of parts.

    Yields
    ------
    array_like
        Dataset parts.

    Raises
    ------
    AttributeError
        If parts' `shape` is equal in each dimension.
        If parts' `shape` and dataset's shape are not conformable.

    Examples
    --------
    >>> [ x for x in partition_ignore(np.ones((3, 3), dtype=int), shape=(2, 2)) ]
    [array([[1, 1],
           [1, 1]])]
    """
    for part in partition(x, shape, shift=0):
        if part.shape == shape:
            yield part

def partition_shrink(x, shape, min_width):
    """Partition with shrinking shape boundary condition.

    Parameters
    ----------
    x array_like
        Dataset of arbitrary dimensionality represented by a *Numpy* array.
    shape : tuple
        Shape of parts.
    min_width : int
        Minimal width of parts' shape.

    Yields
    ------
    array_like
        Dataset parts.

    Raises
    ------
    AttributeError
        If parts' `shape` is equal in each dimension.
        If parts' `shape` and dataset's shape are not conformable.

    Examples
    --------
    >>> data = np.ones((5, 5), dtype=int)
    >>> [ x for x in partition_shrink(data, shape=(3, 3), min_width=2)]
    [array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]]), array([[1, 1],
           [1, 1]]), array([[1, 1],
           [1, 1]]), array([[1, 1],
           [1, 1]])]
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

    Raises
    ------
    KeyError
        If key of an object can not be found in the reference CTM lookup table.

    Examples
    --------
    >>> from bdm import BDM
    >>> bdm = BDM(ndim=1)
    >>> data = np.ones((16, ), dtype=int)
    >>> parts = partition_shrink(data, (12, ), min_width=4)
    >>> [ x for x in lookup(parts, bdm._ctm) ]
    [('111111111111', 1.95207842085224e-08), ('1111', 0.00409810315977953)]
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

    Examples
    --------
    >>> from bdm import BDM
    >>> bdm = BDM(ndim=1)
    >>> data = np.ones((40, ), dtype=int)
    >>> parts = partition_shrink(data, (12, ), min_width=4)
    >>> ctms = lookup(parts, bdm._ctm)
    >>> aggregate(ctms)
    1.5890606234017197
    """
    counter = Counter()
    for key, ctm in ctms:
        counter.update([ (key, ctm) ])
    bdm = 0
    for key, n in counter.items():
        _, ctm = key
        bdm += ctm + np.log2(n)
    return bdm
