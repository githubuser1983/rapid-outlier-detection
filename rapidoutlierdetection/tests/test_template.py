import numpy as np
from numpy.testing import assert_almost_equal

from rapidoutlierdetection import RapidOutlierDetection


def test_rapidoutlierdetection():
    X = np.random.random((100, 10))
    rad = RapidOutlierDetection()
    rad.fit(X)
    assert_almost_equal(rad.predict(X), )
