"""
Meta-features from a model
==========================

In this example, we will show you how to extract meta-features from a
pre-fitted model.
"""

# Load a dataset
import sklearn.tree
from sklearn.datasets import load_iris
from pymfe.mfe import MFE

iris = load_iris()

###############################################################################
# If you want to extract metafeatures from a pre-fitted machine learning model
# (from sklearn package), you can use the `extract_from_model` method without
# needing to use the training data:

# Extract from model
model = sklearn.tree.DecisionTreeClassifier().fit(iris.data, iris.target)
extractor = MFE()
ft = extractor.extract_from_model(model)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

# Extract specific metafeatures from model
extractor = MFE(features=["tree_shape", "nodes_repeated"], summary="histogram")

ft = extractor.extract_from_model(
    model,
    arguments_fit={"verbose": 1},
    arguments_extract={"verbose": 1, "histogram": {"bins": 5}})

print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))
