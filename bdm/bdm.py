"""Block Decomposition Method

`BDM` class provides a top-level interface for configuring an instance
of a block decomposition method as well as running actual computations
approximating algorithmic complexity of given datasets.

Configuration step is necessary for specifying dimensionality of allowed
datasets as well as boundary conditions for block decomposition etc.
"""
import pickle
from pkg_resources import resource_stream
from .stages import split_simple, apply_simple, combine_simple


_ndim2shape = {
    1: (12, ),
    2: (4, 4)
}
_ndim2ctm = {
    1: 'ctm-bin-1d.pickle',
    2: 'ctm-bin-2d.pickle'
}


class BDM:
    """Block decomposition method interface.

    Block decomposition method depends on the type data
    (binary sequences or matrices) as well as boundary conditions.

    Block decomposition method is implemented using the *split-apply-combine*
    pipeline approach. First a dataset is partitioned into parts with dimensions
    appropriate for a selected data dimensionality and corresponding
    reference lookup table of CTM value. Then CTM values for all parts
    are extracted. Finally CTM values are aggregated to a single
    approximation of complexity for the entire dataset.
    This stepwise approach makes the implementation modular,
    so every step can be customized during the configuration of a `BDM` object
    or by subclassing.

    Attributes
    ----------
    ndim : int
        Number of dimensions. Positive integer.
    ctm_shape : tuple or None
        Shape of records in a CTM reference dataset.
    ctm_dname : str
        Name of a reference CTM dataset.
        For now it is mean only for inspection purposes
        (this attribute should not be set and changed).
    split : callable
        Split stage method.
    apply : callable
        Apply stage method.
    combine : callable
        Combine stage method
    """
    def __init__(self, ndim, ctm_shape=None, ctm_dname=None,
                 split=split_simple, apply=apply_simple, combine=combine_simple):
        """Initialization method."""
        self.ndim = ndim
        self.ctm_shape = _ndim2shape[ndim] if ctm_shape is None else ctm_shape
        self.ctm_dname = _ndim2ctm[ndim] if ctm_dname is None else ctm_dname
        with resource_stream(__name__, self.ctm_dname) as stream:
            self._ctm = pickle.load(stream)
        self.split = split
        self.apply = apply
        self.combine = combine

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
        parts = self.split(x, self.ctm_shape)
        ctms = self.apply(parts, self._ctm)
        cmx = self.combine(ctms)
        return cmx
