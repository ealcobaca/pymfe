Contributing with pymfe
=======================

How to contribute
-----------------

The preferred way to contribute to pymfe is to fork the
[main repository](https://github.com/ealcobaca/pymfe) on GitHub:

1. Fork the [project repository](https://github.com/ealcobaca/pymfe):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

        $ git clone git@github.com:YourLogin/pymfe.git
        $ cd pymfe

3. Create a branch to hold your changes.
Never work in the ``master`` branch!

        $ git checkout -b my-feature

4. Work on your copy using Git to do the version control.
When you're done editing, do:

        $ git add modified_files
        $ git commit -m "Add a simple message explaining your modifications."

   to record your changes in Git, then push them to GitHub with:

        $ git push -u origin my-feature

Finally, go to the web page of your fork of the pymfe repo,
and click 'Pull request' to send your changes to the maintainers for
review. This will send an email to the Collaborators.

(If any of the above seems like magic to you, then look up the
[Git documentation](https://git-scm.com/documentation) on the web.)

Contributing Pull Requests
--------------------------

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

-  Follow the
   [coding-guidelines](http://scikit-learn.org/dev/developers/contributing.html#coding-guidelines)
   as for scikit-learn.

-  When applicable, use the validation tools and other code in the
   `pymfe._internal` submodule.

-  If your pull request addresses an issue, please use the title to describe
   the issue and mention the issue number in the pull request description to
   ensure a link is created to the original issue.

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  Please prefix the title of your pull request with `[MRG]` if the
   contribution is complete and should be subjected to a detailed review.
   Incomplete contributions should be prefixed `[WIP]` to indicate a work
   in progress (and changed to `[MRG]` when it matures). WIPs may be useful
   to indicate you are working on something to avoid duplicated work,
   request a broad review of functionality or API, or seek collaborators.

-  All other tests pass when everything is rebuilt from scratch. On
   Unix-like systems, check with (from the top-level source folder):

        $ make

-  Documentation and high-coverage tests are necessary for enhancements
   to be accepted.

-  At least one paragraph of documentation with links to
   references in the literature (with PDF links when possible) and
   the example.

You can also check for common programming errors with the following
tools:                           
 
-  Code with good unit test coverage (at least 90%), check with:

        $ pip install pytest pytest-cov
        $ pytest tests/ --showlocals -v --cov=pymfe/

-  For avoiding source-code bug and keep quality, check with:

        $ pip install pylint
        $ pylint path/to/module.py -d 'C0103, R0913, R0902, R0914, C0302, R0904, R0801, E1101'

-  Python typing, check with:

        $ pip install mypy
        $ mypy path/to/module.py --ignore-missing-imports
        
We added a Makefile to execute all this command in a simple way:

- For installing all necessary libraries:

        $ make install-dev

- For checking typing, source code style and code quality:

        $ make code-check

- For executing all the tests:

        $ make test-cov

- For executing all the tests:

        $ make all

Report bugs
-----------
We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/ealcobaca/pymfe/issues)
   or [pull requests](https://github.com/ealcobaca/pymfe/pulls).

-  Please ensure all code snippets and error messages are formatted on
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please include your operating system type and version number, as well
   as your Python, scikit-learn, numpy, pandas, and scipy versions. This information
   can be found by runnning the following code snippet:

   ```python
   import platform; print(platform.platform())
   import sys; print("python", sys.version)
   import numpy; print("numPy", numpy.__version__)
   import scipy; print("sciPy", scipy.__version__)
   import sklearn; print("scikit-Learn", sklearn.__version__)
   import pandas; print("pandas", pandas.__version__)
   import patsy; print("patsy", pandas.__version__)
   import pymfe; print("pymfe", pymfe.__version__)
   ```
 - If you wish, you can use a predefined [issue template](https://github.com/ealcobaca/pymfe/issues/new/choose).

Documentation
-------------

We are glad to accept any documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc.
reStructuredText documents live in the source code repository under the
doc/ directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the docs/ directory.
The resulting HTML files will be placed in _build/html/ and are viewable in a web browser.
See the README file in the docs/ directory for more information.

For building the documentation, you will need
[sphinx](http://sphinx-doc.org).

When you are writing documentation, it is essential to keep a good
compromise between mathematical and algorithmic details, and give
intuition to the reader on what the algorithm does. It is best to always
start with a small paragraph with a hand-waving explanation of what the
method does to the data and a figure (coming from an example)
illustrating it.


Notes
------

This guide is adapted from 
[scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md) and 
[imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/CONTRIBUTING.md).
