"""Encoding and decoding of arrays with fixed number of unique symbols.

While computing BDM dataset parts have to be encoded into simple hashable objects
such as strings or integers for efficient lookup of CTM values from reference
datasets.

In case of CTM dataset containing objects with several different dimensionalities
string keys have to be used and this representation is used by
:py:module:`bdm.stages` functions at the moment.

Integer encoding can be used for easy generation of objects
of fixed dimensionality as each such object using a fixed,
finite alphabet of symbols can be uniquely mapped to an integer code.
"""
from collections import deque
import numpy as np


def array_from_string(x, shape=None, cast_to=int, sep='-'):
    """Make array from string code.

    Parameters
    ----------
    x : str
        String code.
    shape : tuple or None
        Desired shape of the output array.
        Determined automatically based on `x` is ``None``.
    cast_to : type or None
        Cast array to given type. No casting if ``None``.
        Defaults to integer type.
    sep : str
        Sequence separator.

    Returns
    -------
    array_like
        Array encoded in the string code.

    Examples
    --------
    >>> array_from_string('1010')
    array([1, 0, 1, 0])
    >>> array_from_string('10-00')
    array([[1, 0],
           [0, 0]])
    """
    if sep in x:
        arr = [ list(s) for s in x.split(sep) ]
    else:
        arr = list(x)
    arr = np.array(arr)
    if arr.ndim == 0:
        arr = arr.reshape((1, ))
    if cast_to:
        arr = arr.astype(cast_to)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr

def string_from_array(arr, sep='-'):
    """Encode an array as a string code.

    Parameters
    ----------
    arr : (N, k) array_like
        *Numpy* array.
    sep : str
        Sequence separator.

    Returns
    -------
    str
        String code of an array.

    Examples
    --------
    >>> string_from_array(np.array([1, 0, 0]))
    '100'
    >>> string_from_array(np.array([[1,0], [3,4]]))
    '10-34'
    """
    x = np.apply_along_axis(''.join, arr.ndim - 1, arr.astype(str))
    x = sep.join(np.ravel(x))
    return x

def encode_sequence(seq, base=2):
    """Encode sequence of integer-symbols.

    Parameters
    ----------
    seq : (N, ) array_like
        Sequence of integer symbols represented as 1D *Numpy* array.
    base : int
        Encoding base.
        Should be equal to the number of unique symbols in the alphabet.

    Returns
    -------
    int
        Integer code of a sequence.

    Raises
    ------
    AttributeError
        If `seq` is not 1D.
    TypeError
        If `seq` is not of integer type.
    ValueError
        If `seq` contain values which are negative or beyond the size
        of the alphabet (encoding base).

    Examples
    --------
    >>> encode_sequence(np.array([1, 0, 0]))
    4
    """
    if seq.size == 0:
        return 0
    if seq.ndim != 1:
        raise AttributeError("'seq' has to be a 1D array")
    if seq.dtype != np.int:
        raise TypeError("'seq' has to be of integer dtype")
    if not (seq >= 0).all():
        raise ValueError("'seq' has to consist of non-negative integers")
    proper_values = np.arange(base)
    if not np.isin(seq, proper_values).all():
        raise ValueError(f"There are symbol codes greater than {base-1}")
    code = 0
    for i, x in enumerate(reversed(seq)):
        if x > 0:
            code += x * base**i
    return code

def decode_sequence(code, base=2, min_length=None):
    """Decode sequence from a sequence code.

    Parameters
    ----------
    code : int
        Non-negative integer.
    base : int
        Encoding base.
        Should be equal to the number of unique symbols in the alphabet.
    min_length : int or None
        Minimal number of represented bits.
        Use shortest representation if ``None``.

    Returns
    -------
    array_like
        1D *Numpy* array.

    Examples
    --------
    >>> decode_sequence(4)
    array([1, 0, 0])
    """
    bits = deque()
    while code > 0:
        code, rest = divmod(code, base)
        bits.appendleft(rest)
    n = len(bits)
    if min_length and n < min_length:
        for _ in range(min_length - n):
            bits.appendleft(0)
    return np.array(bits)

def encode_string(x, base=2):
    """Encode sequence-string to integer code.

    Parameters
    ----------
    x : str
        Sequence string.
    Base : int
        Encoding base.

    Returns
    -------
    int
        Integer code for a sequence-string.
    """
    return encode_array(array_from_string(x), base=base)

def decode_string(code, shape, base=2):
    """Decode sequence-string from an integer code.

    Parameters
    ----------
    code : int
        Non-negative integer.
    base : int
        Encoding base.
    min_length : int or None
        Minimal number of represented bits.
        Use shortest representation if ``None``.

    Returns
    -------
    str
        Sequence-string corresponding to an integer code.
    """
    return string_from_array(decode_array(code, shape, base=base))

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

    Returns
    -------
    int
        Integer code of an array.
    """
    seq = np.ravel(x, **kwds)
    return encode_sequence(seq, base=base)

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

    Returns
    -------
    array_like
        *Numpy* array.
    """
    length = np.multiply.reduce(shape)
    seq = decode_sequence(code, base=base, min_length=length)
    if seq.size > length:
        raise ValueError(f"{code} does not encode array of shape {shape}")
    arr = seq.reshape(shape, **kwds)
    return arr
