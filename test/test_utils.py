"""Tests for utility functions."""
import pytest
import numpy as np
from bdm.utils import array_from_string, string_from_array


@pytest.mark.parametrize('x,expected', [
    ('', np.array([])),
    ('0000-1000-0101', np.array([[0,0,0,0], [1,0,0,0], [0,1,0,1]]))
])
def test_array_from_string(x, expected):
    output = array_from_string(x)
    assert (output == expected).all()

@pytest.mark.parametrize('arr,expected', [
    (np.array([]), ''),
    (np.array([[0,0,0,0], [1,0,0,0], [0,1,0,1]]), '0000-1000-0101')
])
def test_string_from_array(arr, expected):
    output = string_from_array(arr)
    assert output == expected
