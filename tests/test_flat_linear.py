import pytest

import pygorpho as pg
import numpy as np

def test_dilate():
    vol = np.zeros((7,7,7))
    vol[3,3,3] = 1

    lineSteps = np.array([[1,0,0],[0,1,0],[0,0,1]])
    lineLens = np.array([3, 4, 5])

    expected = np.zeros((7,7,7))
    expected[2:5,2:6,1:6] = 1

    actual1 = pg.flat.linear_morph(vol, lineSteps, lineLens, pg.DILATE)

    np.testing.assert_equal(actual1, expected)

    actual2 = pg.flat.linear_dilate(vol, lineSteps, lineLens)
    np.testing.assert_equal(actual2, expected)


def test_erode():
    vol = np.ones((7,7,7))
    vol[3,3,3] = 0

    lineSteps = np.array([[1,0,0],[0,1,0],[0,0,1]])
    lineLens = np.array([3, 4, 5])

    expected = np.ones((7,7,7))
    expected[2:5,2:6,1:6] = 0

    actual1 = pg.flat.linear_morph(vol, lineSteps, lineLens, pg.ERODE)
    np.testing.assert_equal(actual1, expected)

    actual2 = pg.flat.linear_erode(vol, lineSteps, lineLens)
    np.testing.assert_equal(actual2, expected)


def test_invalid_op():
    with pytest.raises(AssertionError):
        pg.flat.linear_morph([], [], [], 99)


def test_resize():
    vol = np.ones((3,3))
    line_steps = np.array([1,0,0])
    line_lens = np.ones(1)
    res = pg.flat.linear_dilate(vol, line_steps, line_lens)
    assert res.shape == vol.shape


def test_non_numpy_input():
    vol = [0, 0, 1, 0, 0]
    line_steps = [0, 1, 0]
    line_lens = 3
    expected = [0, 1, 1, 1, 0]
    actual = pg.flat.linear_dilate(vol, line_steps, line_lens)
    np.testing.assert_equal(actual, expected)


def test_valid_dims():
    vol = []
    line_steps = 1
    line_lens = 1
    with pytest.raises(AssertionError):
        pg.flat.linear_dilate(vol, line_steps, line_lens)
