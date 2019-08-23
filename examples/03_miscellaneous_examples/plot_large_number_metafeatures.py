"""
Extracting large number of metafeatures
=======================================

In this example, we will extract all possible metafeatures from the Iris
dataset.
"""

from sklearn.datasets import load_iris
from pymfe.mfe import MFE

# Load a dataset
data = load_iris()
y = data.target
X = data.data


###############################################################################
# Using standard parameters, we will get only a few metafeatures. They are most
# commonly used in the community.
mfe = MFE()
mfe.fit(X, y)
ft = mfe.extract()
print(len(ft[0]))


###############################################################################
# Using the value ``all`` you can extract all available metafeatures. For
# this, set the ``groups`` and ``summary`` with ``all``.
mfe = MFE(groups="all", summary="all")
mfe.fit(X, y)
ft = mfe.extract()
print(len(ft[0]))


###############################################################################
# .. note::
#     Be careful when using all the metafeatures because you can bring to
#     meta-level the curse of dimensionality.
