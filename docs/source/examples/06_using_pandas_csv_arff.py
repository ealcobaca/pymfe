"""
06 - Using Pandas, CSV and ARFF files
=====================================

In this example we will show you how to use Pandas, CSV and ARFF in pymfe.
"""

###############################################################################
# Necessary imports
# -----------------
import pandas as pd
import numpy as np
from numpy import genfromtxt
from pymfe.mfe import MFE
import csv
import arff

###############################################################################
# Pandas
# ------
# Generating fake data

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
y = pd.DataFrame({
    'class': np.repeat(['C1', 'C2'], sample_size/2)
}).astype(str)

# Exploring data characteristic
print("X shape --> ", X.shape)
print("y shape --> ", y.shape)
print("classes --> ", np.unique(y))
print("X dtypes --> ", X.dtypes)
print("y dtypes --> ", y.dtypes)

# For extracting meta-features, you should send X and y as a sequence, like
# ``numpy`` array or python ``list``.
# It is easy to make this using ``pandas``:

mfe = MFE(
    groups=["general", "statistical", "info-theory"],
    random_state=42
)
mfe.fit(X.values, y.values)
ft = mfe.extract(cat_cols='auto', suppress_warnings=True)
print(ft)

# df = X.join(y).to_csv('data/data.csv', index=False,
#                       quoting=csv.QUOTE_NONNUMERIC, quotechar="'")
df = X.join(y).to_csv('data/data.csv', index=False)


###############################################################################
# Pandas CSV
# ----------
# Getting CSV data
df = pd.read_csv('data/data.csv')
X, y = df.drop('class', axis=1), df['class']
Xcsv = X
# Exploring data characteristic
print("X shape --> ", X.shape)
print("y shape --> ", y.shape)
print("classes --> ", np.unique(y))
print("X dtypes --> ", X.dtypes)
print("y dtypes --> ", y.dtypes)

# For extracting meta-features, you should send X and y as a sequence, like
# ``numpy`` array or python ``list``.
# It is easy to make this using ``pandas``:

mfe = MFE(
    groups=["general", "statistical", "info-theory"],
    random_state=42
)
mfe.fit(X.values, y.values)
ft = mfe.extract(cat_cols='auto', suppress_warnings=True)
print(ft)

###############################################################################
# ARFF
# ----
# Getting ARFF data:

data = arff.load(open('data/data.arff', 'r'))['data']
X = [i[:4] for i in data]
y = [i[-1] for i in data]

# Exploring data characteristic
print("X shape --> ", len(X))
print("y shape --> ", len(y))
print("classes --> ", np.unique(y))
print("X dtypes --> ", type(X))
print("y dtypes --> ", type(y))

# For extracting meta-features, you should send X and y as a sequence, like
# ``numpy`` array or python ``list``.
# It is directly:

mfe = MFE(
    groups=["general", "statistical", "info-theory"],
    random_state=42
)
mfe.fit(X, y)
ft = mfe.extract(cat_cols='auto', suppress_warnings=True)
print(ft)

# Or
# Getting ARFF data:

classid = 4
data = arff.load(open('data/data.arff', 'r'), encode_nominal=True)
cat_idx = [n for n, i in enumerate(data['attributes'][:classid])
           if isinstance(i[1], list)]
data = np.array(data['data'])
X = data[:, :classid]
y = data[:, classid]
Xarff = X
yarff = y

# Exploring data characteristic
print("X shape --> ", len(X))
print("y shape --> ", len(y))
print("classes --> ", np.unique(y))
print("X dtypes --> ", type(X))
print("y dtypes --> ", type(y))

# For extracting meta-features, you should send X and y as a sequence, like
# ``numpy`` array or python ``list``.
# It is directly:

mfe = MFE(
    groups=["general", "statistical", "info-theory"],
    random_state=42
)
mfe.fit(X, y, cat_cols=cat_idx)
ft = mfe.extract(suppress_warnings=True)
print(ft)
