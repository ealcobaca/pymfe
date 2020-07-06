Using PyMFE
###########
Extracting metafeatures with PyMFE is easy.                                     
 
The simplest way to extract meta-features is by instantiating the `MFE` class.
It computes five meta-features groups by default using mean and standard
deviation as summary functions:  General, Statistical, Information-theoretic,
Model-based, and Landmarking. The `fit` method can be called by passing the `X`
and `y`. Then the `extract` method is used to extract the related measures.
A simple example using `pymfe` for supervised tasks is given next::

   # Load a dataset
   from sklearn.datasets import load_iris
   from pymfe.mfe import MFE

   data = load_iris()
   y = data.target
   X = data.data

   # Extract default measures
   mfe = MFE()
   mfe.fit(X, y)
   ft = mfe.extract()
   print(ft)

   # Extract general, statistical and information-theoretic measures
   mfe = MFE(groups=["general", "statistical", "info-theory"])
   mfe.fit(X, y)
   ft = mfe.extract()
   print(ft)

For more examples see :ref:`sphx_glr_auto_examples`.
