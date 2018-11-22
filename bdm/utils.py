"""Utility functions."""
from . import _ctm_datasets


def list_ctm_datasets():
    """Get a list of available precomputed CTM datasets.

    Examples
    --------
    >>> list_ctm_datasets()
    ['ctm-b2-d12.pickle', 'ctm-b2-d4x4.pickle']
    """
    return _ctm_datasets
