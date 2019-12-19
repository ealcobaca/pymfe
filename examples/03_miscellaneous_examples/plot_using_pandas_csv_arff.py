"""
Using Pandas, CSV and ARFF files
=====================================

In this example we will show you how to use Pandas, CSV and ARFF in PyMFE.
"""

# Necessary imports
import pandas as pd
import numpy as np
from numpy import genfromtxt
from pymfe.mfe import MFE
import csv
import arff

###############################################################################
# Pandas
# ------
# Generating synthetic dataset
np.random.seed(42)

sample_size = 150
numeric = pd.DataFrame({
    'num1': np.random.randint(0, 100, size=sample_size),
    'num2': np.random.randint(0, 100, size=sample_size)
})
categoric = pd.DataFrame({
    'cat1': np.repeat(('cat1-1', 'cat1-2'), sample_size/2),
    'cat2': np.repeat(('cat2-1', 'cat2-2', 'cat2-3'), sample_size/3)
})
X = numeric.join(categoric)
y = pd.Series(np.repeat(['C1', 'C2'], sample_size/2))

###############################################################################
# Exploring characteristics of the data
print("X shape --> ", X.shape)
print("y shape --> ", y.shape)
print("classes --> ", np.unique(y.values))
print("X dtypes --> \n", X.dtypes)
print("y dtypes --> ", y.dtypes)

###############################################################################
# For extracting meta-features, you should send ``X`` and ``y`` as a sequence,
# like numpy array or Python list.
# It is easy to make this using pandas:
mfe = MFE(
    groups=["general", "statistical", "info-theory"],
    random_state=42
)
mfe.fit(X.values, y.values)
ft = mfe.extract(cat_cols='auto', suppress_warnings=True)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))


###############################################################################
# Pandas CSV
# ----------
# Getting data from CSV format
df = pd.read_csv('../data/data.csv')
X, y = df.drop('class', axis=1), df['class']

###############################################################################
# Exploring characteristics of the data
print("X shape --> ", X.shape)
print("y shape --> ", y.shape)
print("classes --> ", np.unique(y))
print("X dtypes --> \n", X.dtypes)
print("y dtypes --> ", y.dtypes)

###############################################################################
# For extracting meta-features, you should send ``X`` and ``y`` as a sequence,
# like numpy array or Python list.
# It is easy to make this using pandas:
mfe = MFE(
    groups=["general", "statistical", "info-theory"],
    random_state=42
)
mfe.fit(X.values, y.values)
ft = mfe.extract(cat_cols='auto', suppress_warnings=True)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))


###############################################################################
# ARFF
# ----
# Getting data from ARFF format:
data = arff.load(open('../data/data.arff', 'r'))['data']
X = [i[:4] for i in data]
y = [i[-1] for i in data]

###############################################################################
# Exploring characteristics of the data
print("X shape --> ", len(X))
print("y shape --> ", len(y))
print("classes --> ", np.unique(y))
print("X dtypes --> ", type(X))
print("y dtypes --> ", type(y))

###############################################################################
# For extracting meta-features, you should send ``X`` and ``y`` as a sequence,
# like numpy array or Python list.
# You can do this directly:
mfe = MFE(
    groups=["general", "statistical", "info-theory"],
    random_state=42
)
mfe.fit(X, y)
ft = mfe.extract(cat_cols='auto', suppress_warnings=True)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))

###############################################################################
# As a final example, we do not use the automatic detection of feature type
# here. We use the ids provided by the liac-arff package.
classid = 4
data = arff.load(open('../data/data.arff', 'r'), encode_nominal=True)
cat_cols = [n for n, i in enumerate(data['attributes'][:classid])
            if isinstance(i[1], list)]
data = np.array(data['data'])
X = data[:, :classid]
y = data[:, classid]

###############################################################################
# Exploring characteristics of the data
print("X shape --> ", len(X))
print("y shape --> ", len(y))
print("classes --> ", np.unique(y))
print("X dtypes --> ", type(X))
print("y dtypes --> ", type(y))

###############################################################################
# For extracting meta-features, you should send ``X`` and ``y`` as a sequence,
# like numpy array or python list.
mfe = MFE(
    groups=["general", "statistical", "info-theory"],
    random_state=42
)
mfe.fit(X, y, cat_cols=cat_cols)
ft = mfe.extract(suppress_warnings=True)
print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))
