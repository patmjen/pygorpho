import pytest

import pygorpho as pg
import numpy as np


def test_dilate():
    vol = np.zeros((7,7,7))
    vol[3,3,3] = 1

    strel = np.ones((3,4,5))

    expected = np.ones_like(vol)
    expected[2:5,2:6,1:6] = 2

    actual1 = pg.gen.dilate_erode(vol, strel, pg.DILATE)
    np.testing.assert_equal(actual1, expected)

    actual2 = pg.gen.dilate(vol, strel)
    np.testing.assert_equal(actual2, expected)


def test_erode():
    vol = np.ones((7,7,7))
    vol[3,3,3] = 0

    strel = np.ones((3,4,5))

    expected = np.zeros_like(vol)
    expected[2:5,2:6,1:6] = -1

    actual1 = pg.gen.dilate_erode(vol, strel, pg.ERODE)
    np.testing.assert_equal(actual1, expected)

    actual2 = pg.gen.erode(vol, strel)
    np.testing.assert_equal(actual2, expected)
