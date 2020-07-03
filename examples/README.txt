The PyMFE example gallery
=========================

In this gallery, we will show a set of examples to help you to use this package and guide you on the meta-feature extraction process. 

In the Meta-learning (MtL) literature, meta-features are measures used to characterize data sets and/or their relations with algorithm bias.
According to Brazdil et al. (2008), "Meta-learning is the study of principled methods that exploit meta-knowledge to obtain efficient models and solutions by adapting the machine learning and data mining process".

Meta-features are used in MtL and AutoML tasks in general, to represent/understand a dataset,  to understanding a learning bias, to create machine learning (or data mining) recommendations systems, and to create surrogates models, to name a few.

Pinto et al. (2016) and Rivolli et al. (2018) defined a meta-feature as follows.
Let <img src="https://render.githubusercontent.com/render/math?math=D \in \mathcal{D}"> be a dataset,
<img src="https://render.githubusercontent.com/render/math?math=m\colon \mathcal{D} \to \mathbb{R}^{k'}"> be a characterization measure,
and <img src="https://render.githubusercontent.com/render/math?math=\sigma\colon \mathbb{R}^{k'} \to \mathbb{R}^{k}"> be a summarization function.
Both <img src="https://render.githubusercontent.com/render/math?math=m"> and 
<img src="https://render.githubusercontent.com/render/math?math=\sigma"> have also hyperparameters associated,
<img src="https://render.githubusercontent.com/render/math?math=h_m"> and
<img src="https://render.githubusercontent.com/render/math?math=h_\sigma"> respectively.
Thus, a meta-feature <img src="https://render.githubusercontent.com/render/math?math=f\colon \mathcal{D} \to \mathbb{R}^{k}"> for a given dataset <img src="https://render.githubusercontent.com/render/math?math=D"> is:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=f\big(D\big) = \sigma\big(m(D,h_m), h_\sigma\big)">.
</p>

The measure <img src="https://render.githubusercontent.com/render/math?math=m"> can extract more than one value from each data set, i.e., <img src="https://render.githubusercontent.com/render/math?math=k'"> can vary according to
<img src="https://render.githubusercontent.com/render/math?math=D">, which can be mapped to a vector of fixed length
<img src="https://render.githubusercontent.com/render/math?math=k"> using a summarization function
<img src="https://render.githubusercontent.com/render/math?math=\sigma">.

In this package, We provided the following meta-features groups:
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
