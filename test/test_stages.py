"""Unit tests for BDM stage functions."""
import pytest
import numpy as np
from bdm.stages import partition_ignore_leftover


@pytest.mark.parametrize('x,shape,expected', [
    (np.ones((2, 2)), (2, 2), [ np.ones((2, 2)) ]),
    (np.ones((5, 5)), (2, 2), [
        np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))
    ])
])
def test_partition_ignore_leftover(x, shape, expected):
    output = [ x for x in partition_ignore_leftover(x, shape) ]
    assert all([ (o == e).all() for o, e in zip(output, expected) ])
