"""
02 - Select specific measures and summaries
===========================================

To customize the measure extraction, is necessary to use the feature
and summary attribute. For instance, ``info-theo and`` and ``statistical``
compute the information theoretical and the statistical measures,
respectively. The following examples illustrate how to run specific measues
and summaries from them:
"""


# Load a dataset
from sklearn.datasets import load_iris
from pymfe.mfe import MFE

data = load_iris()
y = data.target
X = data.data


###############################################################################
# Select specific measures and summaries for ``statistical``
# -------------------------------------------------------------------
#
# Extracting three information theoretical measures.

mfe = MFE(features=["attr_ent", "joint_ent"],
          summary=["median", "min", "max"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)


###############################################################################
# Select specific measures and summaries for ``info-theo``
# ------------------------------------------------------------------
#
# Extracting two statistical measures.

mfe = MFE(features=["can_cor", "cor", "iq_range"],
          summary=["median", "min", "max"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)


###############################################################################
# Select specific measures for both ``info-theo`` and ``statistical``
# --------------------------------------------------------------------
#
# Extracting five measures.

mfe = MFE(features=["attr_ent", "joint_ent", "can_cor", "cor", "iq_range"],
          summary=["median", "min", "max"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)
