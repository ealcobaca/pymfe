"""
Working with the results
========================

In this example, we will show you how to work with the results of metafeatures
extraction.
"""

from sklearn.datasets import load_iris
from pymfe.mfe import MFE

data = load_iris()
y = data.target
X = data.data

###############################################################################
# Parsing subset of metafeaure
# ----------------------------
# After extracting metafeatures, parse a subset of interest from the results.

model = MFE(groups=["relative", "general", "model-based"], measure_time="avg")
model.fit(X, y)
ft = model.extract()
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

###############################################################################
# From the extract output, parse only the 'general' metafeatures
ft_general = model.parse_by_group("general", ft)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft_general[0],
                                                           ft_general[1])))

###############################################################################
# Actually, you can parse by various groups at once. In this case, the selected
# metafeatures must be from one of the given groups.
ft_subset = model.parse_by_group(["general", "model-based"], ft)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft_subset[0],
                                                           ft_subset[1])))

###############################################################################
# Maybe an uncommon scenario, given that the user already have instantiated
# some MFE model to extract the metafeatures, but actually there's no need to
# instantiate a MFE model to parse the results.
ft_subset = MFE.parse_by_group(["general", "model-based"], ft)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft_subset[0],
                                                           ft_subset[1])))
