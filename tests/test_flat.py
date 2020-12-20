import pytest

import pygorpho as pg
import numpy as np

def test_dilate():
    vol = np.zeros((7,7,7))
    vol[3,3,3] = 1

    strel = np.full((3,4,5), True, dtype=bool)

    expected = np.zeros_like(vol)
    expected[2:5,2:6,1:6] = 1

    actual1 = pg.flat.morph(vol, strel, pg.DILATE)
    np.testing.assert_equal(actual1, expected)

    actual2 = pg.flat.dilate(vol, strel)
    np.testing.assert_equal(actual2, expected)


def test_erode():
    vol = np.ones((7,7,7))
    vol[3,3,3] = 0

    strel = np.full((3,4,5), True, dtype=bool)

    expected = np.ones_like(vol)
    expected[2:5,2:6,1:6] = 0

    actual1 = pg.flat.morph(vol, strel, pg.ERODE)
    np.testing.assert_equal(actual1, expected)

    actual2 = pg.flat.erode(vol, strel)
    np.testing.assert_equal(actual2, expected)


def test_open():
    vol = np.zeros((7,7,7))
    vol[3:5,3:5,3:5] = 1  # 2 x 2 x 2 box

    strel = np.full((3,3,3), True, dtype=bool)

    expected = np.zeros_like(vol)

    actual1 = pg.flat.morph(vol, strel, pg.OPEN)
    np.testing.assert_equal(actual1, expected)

    actual2 = pg.flat.open(vol, strel)
    np.testing.assert_equal(actual2, expected)


def test_close():
    vol = np.ones((7,7,7))
    vol[3:5,3:5,3:5] = 0  # 2 x 2 x 2 box

    strel = np.full((3,3,3), True, dtype=bool)

    expected = np.ones_like(vol)

    actual1 = pg.flat.morph(vol, strel, pg.CLOSE)
    np.testing.assert_equal(actual1, expected)

    actual2 = pg.flat.close(vol, strel)
    np.testing.assert_equal(actual2, expected)


def test_tophat():
    vol = np.zeros((7,7,7))
    vol[3:5,3:5,3:5] = 1  # 2 x 2 x 2 box

    strel = np.full((3,3,3), True, dtype=bool)

    expected = np.copy(vol)

    actual1 = pg.flat.morph(vol, strel, pg.TOPHAT)
    np.testing.assert_equal(actual1, expected)

    actual2 = pg.flat.tophat(vol, strel)
    np.testing.assert_equal(actual2, expected)


def test_bothat():
    vol = np.ones((7,7,7))
    vol[3:5,3:5,3:5] = 0  # 2 x 2 x 2 box

    strel = np.full((3,3,3), True, dtype=bool)

    expected = 1 - np.copy(vol)

    actual1 = pg.flat.morph(vol, strel, pg.BOTHAT)
    np.testing.assert_equal(actual1, expected)

    actual2 = pg.flat.bothat(vol, strel)
    np.testing.assert_equal(actual2, expected)


def test_bool_conversion():
    strel = np.ones((3,4,5))

    vol_dilate = np.zeros((7,7,7))
    vol_dilate[3,3,3] = 1

    expected_dilate = np.zeros((7,7,7))
    expected_dilate[2:5,2:6,1:6] = 1

    actual_dilate = pg.flat.dilate(vol_dilate, strel)
    np.testing.assert_equal(actual_dilate, expected_dilate)


def test_invalid_op():
    with pytest.raises(AssertionError):
        pg.flat.morph([], [], 99)


def test_resize():
    vol = np.ones((3,3))
    strel = np.ones(1)
    res = pg.flat.dilate(vol, strel)
    assert res.shape == vol.shape


def test_non_numpy_input():
    vol = [0, 0, 1, 0, 0]
    strel = [1, 1, 1]
    expected = [0, 1, 1, 1, 0]
    actual = pg.flat.dilate(vol, strel)
    np.testing.assert_equal(actual, expected)
