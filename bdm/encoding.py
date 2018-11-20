"""Encoding and decoding of arrays with fixed number of unique symbols.

Each symbol (from a fixed and finite alphabet) has to be mapped to an integer
starting from 0. This allows to construct a 1-to-1 mapping between any array
and the set of non-negative integers.

This technique is useful for compression of CTM datasets and makes it
easier to specify simple input data for unit tests etc.
"""
import numpy as np


def encode_sequence(seq, base=2):
    """Encode sequence of integer-symbols.

    Parameters
    ----------
    seq : (N, ) array_like
        Sequence of integer symbols represented as 1D *Numpy* array.
    base : int
        Encoding base.
        Should be equal to the number of unique symbols in the alphabet.
    """
    # TODO: this should be rewritten as a numpy ufunc and piped into reduce
    if seq.size == 0:
        return 0
    if seq.ndim != 1:
        raise AttributeError("'seq' has to be a 1D array")
    if seq.dtype != np.int:
        raise TypeError("'seq' has to be of integer dtype")
    if not (seq >= 0).all():
        raise ValueError("'seq' has to consist of non-negative integers")
    proper_valeus = np.arange(base)
    if not np.isin(seq, proper_valeus).all():
        raise ValueError(f"There are symbol codes greater than {base-1}")
    code = 0
    for i, x in enumerate(seq):
        if x > 0:
            code += x * base**i
    return code

def encode_array(x, base=2, **kwds):
    """Encode array of integer-symbols.

    Parameters
    ----------
    x : (N, k) array_like
        Array of integer symbols.
    base : int
        Encoding base.
    **kwds :
        Keyword arguments passed to :py:func:`numpy.ravel`.
    """
    seq = np.ravel(x, **kwds)
    return encode_sequence(seq, base=base)

def decode_sequence(code, base=2, min_length=None):
    """Decode sequence from a sequence code.

    Parameters
    ----------
    code : int
        Non-negative integer.
    base : int
        Encoding base.
        Should be equal to the number of unique symbols in the alphabet.
    """
    bits = []
    while code > 0:
        code, rest = divmod(code, base)
        bits.append(rest)
    n = len(bits)
    if min_length and n < min_length:
        bits += [ 0 for _ in range(min_length - n) ]
    return np.array(bits)

def decode_array(code, shape, base=2, **kwds):
    """Decode array of integer-symbols from a sequence code.

    Parameters
    ----------
    code : int
        Non-negative integer.
    shape : tuple of ints
        Expected array shape.
    base : int
        Encoding base.
    **kwds :
        Keyword arguments passed to :py:func:`numpy.reshape`.
    """
    length = np.multiply.reduce(shape)
    seq = decode_sequence(code, base=base, min_length=length)
    if seq.size > length:
        raise ValueError(f"{code} does not encode array of shape {shape}")
    arr = seq.reshape(shape, **kwds)
    return arr
