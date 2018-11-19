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

    Returns
    -------
    list
        Ordered sequence of views of parts of the splitted dataset.
    """
    pass

def lookup(parts, ctm):
    """Lookup CTM values for parts in a reference dataset.

    Parameters
    ----------
    parts : sequence
        Ordered sequence of dataset parts.
    ctm : dict
        Reference CTM dataset.

    Returns
    -------
    ndarray
        Ordered 1D sequence of CTM values for parts.
    """
    pass

def aggregate(ctms):
    """Combine CTM of parts into BDM value.

    Parameters
    ----------
    ctms : sequence
        Ordered 1D sequence of CTM values.
    """
    pass
