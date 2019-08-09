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
from pymfe.mfe import MFE

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
    'cat1': np.repeat(('a', 'b'), sample_size/2),
    'cat2': np.repeat(('c', 'd', 'e'), sample_size/3)
})
X = numeric.join(categoric)
y = pd.DataFrame({
    'class': np.repeat([0, 1], sample_size/2)
})

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
# Pandas CSV
# ----------
# Getting CSV data
df = pd.read_csv('data/data.csv')

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

