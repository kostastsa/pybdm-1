"""Utility functions."""
import numpy as np
from .ctmdata import CTM_DATASETS


def list_ctm_datasets():
    """Get a list of available precomputed CTM datasets.

    Examples
    --------
    >>> list_ctm_datasets()
    ['ctm-b2-d12.pickle', 'ctm-b2-d4x4.pickle']
    """
    return CTM_DATASETS

def get_reduced_shape(x, shape, shift=0, length_only=True):
    """Get shape of a reduced dataset.

    Parameters
    ----------
    x : array_like
        Dataset of arbitrary dimensionality represented as a *Numpy* array.
    shape : tuple
        Shape of the dataset's parts. Has to be symmetric.
    shift : int
        Shift of the sliding window.
        In general, if positive, should not be greater than ``1``.
        Shift by partition shape if not positive.
    length_only : bool
        Should only the 1D length of the reduced dataset be returned.
        1D length is the total number of dataset parts.

    Returns
    -------
    tuple
        Shape tuple if ``length_only=False``.
    int
        Number of parts if ``length_only=True``.

    Raises
    ------
    AttributeError
        If parts' `shape` is equal in each dimension.
        If parts' `shape` and dataset's shape have different numbers of axes.

    Examples
    --------
    >>> x = np.ones((5, 5))
    >>> get_reduced_shape(x, (2, 2), length_only=False)
    (3, 3)
    >>> get_reduced_shape(x, (2, 2), length_only=True)
    9
    """
    if len(set(shape)) != 1:
        raise AttributeError(f"Partition shape is not symmetric {shape}")
    if len(shape) != x.ndim:
        x = x.squeeze()
        if len(shape) != x.ndim:
            raise AttributeError("Dataset and parts have different numbers of axes")
    if shift <= 0:
        r_shape = tuple(int(np.ceil(x / p)) for x, p in zip(x.shape, shape))
    else:
        r_shape = tuple(int(x-p+1) for x, p in zip(x.shape, shape))
    if length_only:
        return int(np.multiply.reduce(r_shape))
    return r_shape

def get_reduced_idx(i, shape):
    """Get index of a part in a reduced representation from a part's number.

    Parameters
    ----------
    i : int
        Part number.
    shape : tuple
        Shape of a reduced dataset.

    Returns
    -------
    tuple
        Index of a part in a reduced dataset.

    Examples
    --------
    >>> get_reduced_idx(5, (2, 2, 2))
    (1, 0, 1)
    >>> get_reduced_idx(2, (1, 4))
    (0, 2)
    """
    K = len(shape)
    r_idx = tuple(
        (i % int(np.multiply.reduce(shape[k:K]))) //
        int(np.multiply.reduce(shape[(k+1):K]))
        for k in range(K)
    )
    return r_idx
