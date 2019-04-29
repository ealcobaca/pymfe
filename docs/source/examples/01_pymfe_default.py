"""
01 - Basic of meta-features extraction
======================================

This example show how to extract meta-features using standard configuration.
"""


###############################################################################
# Extracting meta-features
# ------------------------
#
# The standard way to extract meta-features is using the MFE class.
# The parameters are the dataset and the group of measures to be extracted.
# By default, the method extract all the measures. For instance:

from sklearn.datasets import load_iris
from pymfe.mfe import MFE

# Load a dataset
data = load_iris()
y = data.target
X = data.data

# Extract all measures
mfe = MFE()
mfe.fit(X, y)
ft = mfe.extract()
print(ft)

# Extract general, statistical and information-theoretic measures
mfe = MFE(groups=["general", "statistical", "info-theory"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)


###############################################################################
# Changing summarization function
# -------------------------------
#
# Several measures return more than one value. To aggregate them, post
# processing methods can be used. It is possible to compute min, max, mean,
# median, kurtosis, standard deviation, among others. The default methods are
# the mean and the sd. For instance:
#

# Compute all measures using min, median and max
mfe = MFE(summary=["min", "median", "max"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)

# Compute all measures using quantile
mfe = MFE(summary=["quantiles"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)
