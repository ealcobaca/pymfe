"""
Extracting meta-features from unsupervised learning
===================================================

In this example we will show you how to extract meta-features from unsupervised
machine learning tasks.
"""

# Load a dataset
from sklearn.datasets import load_iris
from pymfe.mfe import MFE

data = load_iris()
y = data.target
X = data.data

###############################################################################
#
# You can simply omit the target attribute for unsupervised tasks while
# fitting the data into the MFE model. The `pymfe` package automatically finds
# and extracts only the metafeatures suitable for this type of task.

# Extract default unsupervised measures
mfe = MFE()
mfe.fit(X)
ft = mfe.extract()
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

# Extract all available unsupervised measures
mfe = MFE(groups="all")
mfe.fit(X)
ft = mfe.extract()
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))
