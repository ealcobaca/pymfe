What is new on PyMFE package?
#############################
The PyMFE releases are available in PyPI_ and GitHub_.

.. _PyPI: https://pypi.org/project/pymfe/
.. _GitHub: https://github.com/ealcobaca/pymfe/releases


Version 0.3.0 (Available on PyPI)
---------------------------------
* Metafeature extraction with confidence intervals

* Pydoc fixes and package documentation/code consistency improvements

  * Reformatted 'model-based' group metafeature extraction methods arguments to
    a consistent format (all model-based metafeatures now receive a single
    mandatory argument 'dt_model', and all other arguments are optional
    arguments from precomputations.) Now it is much easier to use those
    methods directly without the main class (mfe) filter, if desired.

  * Now accepting user custom arguments in precomputation methods.

  * Added 'extract_from_model' MFE method, making easy to extract model-based
    metafeatures from a pre-fitted model without using the training data.

* Memory issues

  * Now handling memory errors in precomputations, postcomputations and
    metafeature extraction as a regular exception.

* Categorical attributes one-hot encoding option

  * Added option to encode categorical attributes using one-hot encoding
    instead of the current gray encoding.

* New nan-resilient summary functions

  * All summary functions now can be calculated ignoring 'nan' values, using
    its nan-resilient version.

* Online documentation improvement


Version 0.2.0 (Available on PyPI)
---------------------------------
* New meta-feature groups

  * Complexity

  * Itemset

  * Concept

* New feature in MFE to list meta-feature description and references

* Dev class update

* Integration, system tests, tests updates

* Old module reviews

* Docstring improvement

* Online documentation improvement

* Clustering group updated

* Landmarking group updated

* Statistical group updated


Version 0.1.1 (Available on PyPI)
---------------------------------
* Bugs solved

   * False positive of mypy fixed

   * Contributing link now is working

* We added a note about how to add a new meta-feature

* Modified 'verbosity' (from 'extract' method) argument type from boolean to
  integer. Now the user can choose the desired level of verbosity.
  Verbosity = 1 means that a progress bar will be shown during the meta-feature
  extraction process. Verbosity = 2 maintains all the previous verbose messages
  (i.e., it logs every "extract" step) plus additional information about the
  current percentage of progress done so far.


Version 0.1.0 (Available on PyPI)
---------------------------------
* Meta-feature groups available

   * Relative landmarking

   * Clustering-based

   * Relative subsampling landmarking

* Makefile to help developers

* New Functionalities

   * Now you can list available groups

   * Now you can list available meta-features

* Documentation

   * New examples

   * New README

* Bugs

   * Problems in parse categoric meta-features solved

   * Categorization of attributes with constant values solved

* Test

   * Several new tests added


Version 0.0.3 (Available on PyPI)
---------------------------------
* Documentation improvement
  
* Setup improvement


Initial Release
---------------
* Meta-feature groups available:

  * Simple

  * Statistical

  * Information-theoretic

  * Model-based

  * Landmarking


