The pymfe example gallery
=========================

In this gallery, we will show a set of examples to help you to use this package and guide you on the meta-feature extraction process. 


Extracts meta-features from datasets to support the design of recommendation systems based on Meta-Learning (MtL). The meta-features, also called characterization measures, are able to characterize the complexity of datasets and to provide estimates of algorithm performance. The package contains not only the standard, but also more recent characterization measures. By making available a large set of meta-feature extraction functions, this package allows a comprehensive data characterization, a deep data exploration and a large number of MtL-based data analysis.

Measures
--------

In MtL, meta-features are designed to extract general properties able to characterize datasets. The meta-feature values should provide relevant evidences about the performance of algorithms, allowing the design of MtL-based recommendation systems. Thus, these measures must be able to predict, with a low computational cost, the performance of the algorithms under evaluation. In this package, the meta-feature measures are divided into 11 groups:

- **General**: General information related to the dataset, also known as simple measures, such as the number of instances, attributes and classes.
- **Statistical**: Standard statistical measures to describe the numerical properties of data distribution.
- **Information-theoretic**: Particularly appropriate to describe discrete (categorical) attributes and their relationship with the classes.
- **Model-based**: Measures designed to extract characteristics from simple machine learning models.
- **Landmarking**: Performance of simple and efficient learning algorithms.
- **Relative Landmarking**: Relative performance of simple and efficient learning algorithms.
- **Subsampling Landmarking**: Performance of simple and efficient learning algorithms from a subsample of the dataset.
- **Clustering**: Clustering measures extract information about dataset based on external validation indexes.
- **Concept**: Estimate the variability of class labels among examples and the examples density.
- **Itemset**: Compute the correlation between binary attributes.
- **Complexity**: Estimate the difficulty in separating the data points into their expected classes.

Below is a gallery of examples:
