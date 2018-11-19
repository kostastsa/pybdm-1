"""Tests for `bdm` module."""
# pylint: disable=W0621
import pytest
import numpy as np
from bdm import BDM


@pytest.fixture(scope='session')
def bdmobj():
    return BDM(ndim=2)

def _str2array(x):
    arr = np.array([ x for x in map(list, x.split('-')) ]).astype(int)
    return arr


class TestBDM:

    @pytest.mark.parametrize('x,expected', [
        (_str2array('00000001-00010001-11110001-10000001'), 56.3596),
        (_str2array('00001111-00011111-11111111-10011111'), 54.629000000000005),
        (_str2array('00001000-00011000-11111000-10001000'), 56.3596),
        (_str2array('00000000-00010000-11110000-10100000'), 50.7376)
    ])
    def test_complexity(self, bdmobj, x, expected):
        output = bdmobj.complexity(x)
        assert output == expected
