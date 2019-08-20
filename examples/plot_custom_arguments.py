"""
05 - Customizing measures arguments
===================================

In this example we will show you how to custorize the measures.
"""


# Load a dataset
from sklearn.datasets import load_iris
from pymfe.mfe import MFE

data = load_iris()
y = data.target
X = data.data


###############################################################################
# Custom Arguments
# ----------------
#
# It is possible to pass custom arguments to every meta-feature using pymfe
# extract method kwargs. The keywords must be the target meta-feature name, and
# the value must be a dictionary in the format {argument: value}, i.e., each
# key in the dictionary is a target argument with its respective value. In the
# example below, the extraction of metafeatures ``min`` and ``max`` happens as
# usual, but the meta-features ``sd``, ``nr_norm`` and ``nr_cor_attr`` will
# receive user custom argument values, which will interfere in each metafeature
# result.

# Extract measures with custom user arguments
mfe = MFE(features=["sd", "nr_norm", "nr_cor_attr", "min", "max"])
mfe.fit(X, y)
ft = mfe.extract(
    sd={"ddof": 0},
    nr_norm={"method": "all", "failure": "hard", "threshold": 0.025},
    nr_cor_attr={"threshold": 0.6},
)
print("\n".join("{:50} {:50}".format(x, y) for x, y in zip(ft[0], ft[1])))
