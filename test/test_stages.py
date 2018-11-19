"""Unit tests for BDM stage functions."""
# pylint: disable=E1101,W0621
import os
import pickle
import pytest
import numpy as np
import bdm.resources
from bdm.stages import partition_ignore_leftover, lookup


@pytest.fixture(scope='session')
def ctmbin2d():
    """CTM reference dataset for 2D binary matrices."""
    path = os.path.join(bdm.resources.__path__._path[0], 'ctm-bin-2d.pickle')
    with open(path, 'rb') as stream:
        return pickle.load(stream)


@pytest.mark.parametrize('x,shape,expected', [
    (np.ones((2, 2)), (2, 2), [ np.ones((2, 2)) ]),
    (np.ones((5, 5)), (2, 2), [
        np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))
    ])
])
def test_partition_ignore_leftover(x, shape, expected):
    output = [ x for x in partition_ignore_leftover(x, shape) ]
    assert all([ (o == e).all() for o, e in zip(output, expected) ])

@pytest.mark.parametrize('parts,expected', [
    ([ np.ones((4, 4)).astype(int) ], [ ('1111-1111-1111-1111', 22.0067) ]),
])
def test_lookup(parts, ctmbin2d, expected):
    output = [ x for x in lookup(parts, ctmbin2d) ]
    assert output == expected
