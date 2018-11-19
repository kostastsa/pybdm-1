"""Block decomposition stage functions."""


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
    pass

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
