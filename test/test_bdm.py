"""Tests for `bdm` module."""
# pylint: disable=W0621
import os
import pytest
from bdm import BDM
from bdm.utils import array_from_string

_dirpath = os.path.split(__file__)[0]
# Get test input data and expected values
bdm_test_input = []
with open(os.path.join(_dirpath, 'bdm-b2-d4x4-test-input.tsv'), 'r') as stream:
    for line in stream:
        string, bdm = line.strip().split("\t")
        bdm = float(bdm.strip())
        arr = array_from_string(string.strip())
        bdm_test_input.append((arr, bdm))


@pytest.fixture(scope='session')
def bdmobj():
    return BDM(ndim=2)


class TestBDM:

    @pytest.mark.parametrize('x,expected', bdm_test_input)
    def test_complexity(self, bdmobj, x, expected):
        output = bdmobj.complexity(x)
        assert output == expected
