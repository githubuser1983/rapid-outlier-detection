print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from rapidoutlierdetection import RapidOutlierDetection

np.random.seed(42)

# Generate train data
N_normal = 10000
X = 0.3 * np.random.randn(N_normal, 2)
# Generate some abnormal novel observations
N_outliers = 30
X_outliers = np.random.uniform(low=-4, high=4, size=(N_outliers, 2))
X = np.r_[X + 2, X - 2, X_outliers]

# fit the model
clf = RapidOutlierDetection(contamination = 1.0*N_outliers/(2.0*N_normal+N_outliers))
y_pred = clf.fit_predict(X)
y_pred_outliers = y_pred[2*N_normal:]


plt.title("Rapid Outlier Detection")

a = plt.scatter(X[:(2*N_normal), 0], X[:(2*N_normal), 1], c='white',
                edgecolor='k', s=N_outliers)
b = plt.scatter(X[(2*N_normal):, 0], X[(2*N_normal):, 1], c='red',
                edgecolor='k', s=N_outliers)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a, b],
           ["normal observations",
            "abnormal observations"],
           loc="upper left")
plt.show()