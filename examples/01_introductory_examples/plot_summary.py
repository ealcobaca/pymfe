"""
Using Summaries
===============

In this example we will explain the different ways to select summary functions.
"""


# Load a dataset
from sklearn.datasets import load_iris
from pymfe.mfe import MFE

data = load_iris()
y = data.target
X = data.data


###############################################################################
# Summary Methods
# ---------------
#
# Several meta-features generate multiple values and ``mean`` and ``sd`` are
# the standard method to summary these values. In order to increase the
# flexibility, the PyMFE package implemented the summary (or post processing)
# methods to deal with multiple measures values. This method is able to deal
# with descriptive statistic (resulting in a single value) or a distribution
# (resulting in multiple values).
#
# The post processing methods are setted using the parameter summary.
# It is possible to compute min, max, mean, median, kurtosis, standard
# deviation, among others. It will be illustrated in the following examples:

###############################################################################
# Apply several statistical measures as post processing
mfe = MFE(summary=["max", "min", "median", "mean", "var", "sd", "kurtosis",
                   "skewness"])
mfe.fit(X, y)
ft = mfe.extract()
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

###############################################################################
# Apply quantile as post processing method
mfe = MFE(features=["cor"], summary=["quantiles"])
mfe.fit(X, y)
ft = mfe.extract()
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

###############################################################################
# Apply histogram as post processing method
mfe = MFE(features=["cor"], summary=["histogram"])
mfe.fit(X, y)
ft = mfe.extract()
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

###############################################################################
# Get the default values without summarize them
mfe = MFE(features=["cor"], summary=None)
mfe.fit(X, y)
ft = mfe.extract()
print(ft)
