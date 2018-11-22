"""Block Decomposition Method

`BDM` class provides a top-level interface for configuring an instance
of a block decomposition method as well as running actual computations
approximating algorithmic complexity of given datasets.

Configuration step is necessary for specifying dimensionality of allowed
datasets, encoding of reference CTM data as well as
boundary conditions for block decomposition etc.
"""
import pickle
from pkg_resources import resource_stream
from .stages import partition_ignore, lookup, aggregate
from .ctmdata import __name__ as ctmdata_path


_ndim_to_shape = {
    1: (12, ),
    2: (4, 4)
}
_ndim_to_ctm = {
    1: 'ctm-b2-d12.pickle',
    2: 'ctm-b2-d4x4.pickle'
}


class BDM:
    """Block decomposition method interface.

    Block decomposition method depends on the dimensionality of data
    and the length of the symbols alphabet which determines base for encoding
    of reference CTM data. Binary data is encoded in base 2, 3-symbols alphabets
    in base 3 and so forth.

    Block decomposition method is implemented using the *split-apply-combine*
    pipeline approach. First a dataset is partitioned into parts with dimensions
    appropriate for a selected data dimensionality and corresponding
    reference lookup table of CTM value. Then CTM values for all parts
    are extracted. Finally CTM values are aggregated to a single
    approximation of complexity for the entire dataset.
    This stepwise approach makes the implementation modular,
    so every step can be customized during the configuration of a `BDM` object
    or by subclassing.

    Notes
    -----
    Currently CTM reference datasets are computed only for binary sequences
    of length up to 12 and binary 4-by-4 binary matrices.

    Attributes
    ----------
    ndim : int
        Number of dimensions. Positive integer.
    base : int
        Base for encoding of CTM data. Greater or equal to 2.
    ctm_width : int or None
        Width of the sliding window and CTM records.
    ctm_dname : str
        Name of a reference CTM dataset.
        For now it is mean only for inspection purposes
        (this attribute should not be set and changed).
    partition : callable
        Partition stage method.
    lookup : callable
        Lookup stage method.
    aggregate : callable
        Aggregate stage method
    """
    def __init__(self, ndim, base=2, ctm_width=None, ctm_dname=None,
                 partition_func=partition_ignore,
                 lookup_func=lookup, aggregate_func=aggregate):
        """Initialization method."""
        self.ndim = ndim
        self.base = base
        if ctm_width is None:
            self.ctm_shape = _ndim_to_shape[ndim]
        else:
            self.ctm_shape = tuple([ ctm_width for _ in range(ndim) ])
        self.ctm_dname = _ndim_to_ctm[ndim] if ctm_dname is None else ctm_dname
        with resource_stream(ctmdata_path, self.ctm_dname) as stream:
            self._ctm = pickle.load(stream)
        self.partition = partition_func
        self.lookup = lookup_func
        self.aggregate = aggregate_func

    def complexity(self, x):
        """Approximate complexity of a dataset.

        Parameters
        ----------
        x : (N, k) array_like
            Dataset representation as a :py:class:`numpy.ndarray`.

        Returns
        -------
        float
            Approximate algorithmic complexity.
        """
        parts = self.partition(x, self.ctm_shape)
        ctms = self.lookup(parts, self._ctm, base=self.base)
        cmx = self.aggregate(ctms)
        return cmx
