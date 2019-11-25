"""
Listing available metafeatures, groups, and summaries
=====================================================

In this example, we will show you how to list the types of metafeatures,
groups, and summaries available.
"""

from sklearn.datasets import load_iris
from pymfe.mfe import MFE

###############################################################################
# Print all available metafeature groups from the ``pymfe`` package.
model = MFE()
model_groups = model.valid_groups()
print(model_groups)

###############################################################################
# Actually, there's no need to instantiate a model for that
model_groups = MFE.valid_groups()
print(model_groups)

###############################################################################
# Print all available metafeatures from some groups of the ``pymfe`` package
# If no parameter is given (or is 'None'), then all available
# will be returned.
model = MFE()
mtfs_all = model.valid_metafeatures()
print(mtfs_all)

###############################################################################
# Again, there's no need to instantiate a model to invoke this method
mtfs_all = MFE.valid_metafeatures()
print(mtfs_all)

###############################################################################
# You can specify a group name or a collection of group names to
# check their correspondent available metafeatures only
mtfs_landmarking = MFE.valid_metafeatures(groups="landmarking")
print(mtfs_landmarking)

mtfs_subset = MFE.valid_metafeatures(groups=["general", "relative"])
print(mtfs_subset)

###############################################################################
# Print all available summary functions from the ``pymfe`` package
model = MFE()
summaries = model.valid_summary()
print(summaries)

###############################################################################
# Once again, there's no need to instantiate a model to accomplish this
summaries = MFE.valid_summary()
print(summaries)
