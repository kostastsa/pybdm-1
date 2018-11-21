"""Unit tests for encoding/decoding functions."""
import pytest
import numpy as np
from bdm.encoding import encode_sequence, decode_sequence
from bdm.encoding import encode_array, decode_array


@pytest.mark.parametrize('seq,base,expected', [
    (np.array([]), 2, 0),
    (np.array([1, 0, 0, 1]), 2, 9),
    (np.array([1, 0, 0, 1]), 3, 28)
])
def test_encode_sequence(seq, base, expected):
    output = encode_sequence(seq, base=base)
    assert output == expected

@pytest.mark.parametrize('code,base,min_length,expected', [
    (0, 2, None, np.array([])),
    (9, 2, None, np.array([1, 0, 0, 1])),
    (28, 3, None, np.array([1, 0, 0, 1])),
    (20, 4, 5, np.array([0, 1, 1, 0, 0]))
])
def test_decode_sequence(code, base, min_length, expected):
    output = decode_sequence(code, base=base, min_length=min_length)
    assert (output == expected).all()

@pytest.mark.parametrize('x,base,expected', [
    (np.array([]), 7, 0),
    (np.array([0, 1, 1]), 2, 6),
    (np.array([0, 1, 0, 1]), 4, 68)
])
def test_encode_array(x, base, expected):
    output = encode_array(x, base=base)
    assert output == expected

@pytest.mark.parametrize('code,shape,base,expected', [
    (17, (3, 3), 2, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])),
    (7, (2, 2), 3, np.array([[1, 2], [0, 0]]))
])
def test_decode_array(code, shape, base, expected):
    output = decode_array(code, shape, base=base)
    assert (output == expected).all()