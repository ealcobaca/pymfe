# pymfe: Python Meta-Feature Extractor
[![Travis-CI Build Status](https://travis-ci.org/ealcobaca/pymfe.svg?branch=master)](https://travis-ci.org/alcobaca/pymfe)

Extracts meta-features from datasets to support the design of recommendation systems based on Meta-Learning (MtL). The meta-features, also called characterization measures, are able to characterize the complexity of datasets and to provide estimates of algorithm performance. The package contains not only the standard, but also more recent characterization measures. By making available a large set of meta-feature extraction functions, this package allows a comprehensive data characterization, a deep data exploration and a large number of MtL-based data analysis.

## Measures

In MtL, meta-features are designed to extract general properties able to characterize datasets. The meta-feature values should provide relevant evidences about the performance of algorithms, allowing the design of MtL-based recommendation systems. Thus, these measures must be able to predict, with a low computational cost, the performance of the  algorithms under evaluation. In this package, the meta-feature measures are divided into five groups:

* **General**: General information related to the dataset, also known as simple measures, such as number of instances, attributes and classes.
* **Statistical**: Standard statistical measures to describe the numerical properties of a distribution of data.
* **Information-theoretic**: Particularly appropriate to describe discrete (categorical) attributes and their relationship with the classes.
* **Model-based**: Measures designed to extract characteristics like the depth, the shape and size of a Decision Tree (DT) model induced from a dataset.
* **Landmarking**: Represents the performance of simple and efficient learning algorithms.

## Installation

The installation process is similar to other packages available on pip:

```python
pip install pymfe
```

It is possible to install the development version using:

```python
pip install git+https://github.com/ealcobaca/pymfe
```

## Example of use

The simplest way to extract meta-features is using the `metafeatures` method. The method can be called by a symbolic description of the model or by a data frame. The parameters are the dataset and the group of measures to be extracted. The default parameter is extract all the measures. To extract a specific measure, use the function related with the group. A simple example is given next:

```python
# Load a dataset
from sklearn.datasets import load_iris
from pymfe.mfe import MFE

data = load_iris()
y = data.target
X = data.data

# Extract all measures
mfe = MFE()
mfe.fit(X, y)
ft = mfe.extract()
print(ft)

# Extract general, statistical and information-theoretic measures
mfe = MFE(groups=["general", "statistical", "info-theory"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)
```

Several measures return more than one value. To aggregate the returned values, post processed methods can be used. This method can compute min, max, mean, median, kurtosis, standard deviation, among others. The default methods are the `mean` and the `sd`. Next, it is possible to see an example of the use of this method:

```python
## Extract all measures using min, median and max 
mfe = MFE(summary=["min", "median", "max"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)
                          
## Extract all measures using quantile
mfe = MFE(summary=["quantiles"])
mfe.fit(X, y)
ft = mfe.extract()
print(ft)
```

## Developer notes

In the current version, the meta-feature extractor supports only classification problems. The authors plan to extend the package to add clustering and regression measures and to support MtL evaluation measures. For more specific information on how to extract each group of measures, please refer to the functions documentation page and the examples contained therein. For a general overview of the `pymfe` package, please have a look at the associated documentation.

To cite `mfe` in publications use: Alcobaça, E., Siqueira, F., Rivolli, A., Garcia, L. P. F., Oliva, J. T. and de Carvalho, A. C. P. L. F. (2019). MFE: Extracting metafeatures in python and R.

To submit bugs and feature requests, report at [project issues](https://github.com/ealcobaca/pymfe/issues).
