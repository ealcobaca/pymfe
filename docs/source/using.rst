Using pymfe
###########

Extracting metafeatures with pymfe is easy.

The parameters are the measures, the group of measures and the summarization
functions to be extracted. The default parameter is extract all common
measures. The fit function can be called by passing the X and y. The extract
function is used to extract the related measures. See this example::

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

For more examples see :ref:`sphx_glr_auto_examples`.
