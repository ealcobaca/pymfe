"""
Meta-feature confidence interval
================================

In this example, we will show you how to extract meta-features with confidence
interval.
"""

# Load a dataset
import sklearn.tree
from sklearn.datasets import load_iris
from pymfe.mfe import MFE

data = load_iris()
y = data.target
X = data.data

# You can also extract your meta-features with confidence intervals using
# bootstrap. Keep in mind that this method extracts each meta-feature several
# times, and may be very expensive depending mainly on your data and the
# number of meta-feature extract methods called.

# Extract meta-features with confidence interval
mfe = MFE(features=["mean", "nr_cor_attr", "sd", "max"])
mfe.fit(X, y)

ft = mfe.extract_with_confidence(
    sample_num=256,
    confidence=0.99,
    verbose=1,
)

print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))
