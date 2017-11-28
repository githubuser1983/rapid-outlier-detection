import sys
sys.path.insert(1,'/usr/local/lib/python2.7/dist-packages/')

from sklearn.utils.estimator_checks import check_estimator
from rapidoutlierdetection import RapidOutlierDetection


def test_estimator():
    return check_estimator(RapidOutlierDetection)


