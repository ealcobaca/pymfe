"""
Metafeature description
=======================

In this example, we will show you how to list the types of metafeatures,
groups, and summaries available.
"""

from pymfe.mfe import MFE


###############################################################################
# This function shows the description of all metafeatures.
MFE.metafeature_description()

###############################################################################
# You can select a specific group.
MFE.metafeature_description(groups=["general", "statistical"])

###############################################################################
# You can sort the metafeatures by name and group.
MFE.metafeature_description(sort_by_group=True, sort_by_mtf=True)

###############################################################################
# You also can get the table instead of printing it.
MFE.metafeature_description(print_table=False)
