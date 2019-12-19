What is new on pymfe package?
#############################
The pymfe releases are available in PyPI_ and GitHub_.

.. _PyPI: https://pypi.org/project/pymfe/
.. _GitHub: https://github.com/ealcobaca/pymfe/releases

Version 0.2.0
-------------
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


Version 0.1.1
-------------
* Bugs solved

   * False positive of mypy fixed

   * Contributing link now is working

* We added a note about how to add a new meta-feature

* Modified 'verbosity' (from 'extract' method) argument type from boolean to
  integer. Now the user can choose the desired level of verbosity.
  Verbosity = 1 means that a progress bar will be shown during the metafeature
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

   * Now you can list available metafeatures

* Documentation

   * New examples

   * New README

* Bugs

   * Problems in parse categoric metafeatures solved

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

