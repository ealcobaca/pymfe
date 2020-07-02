"""Setup for pymfe package."""
import setuptools
import os
import pymfe


with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()


NAME = 'pymfe'


VERSION = pymfe.__version__


DESCRIPTION = 'Meta-feature Extractor'


LICENSE = 'MIT'


URL = 'https://github.com/ealcobaca/pymfe'


DOWNLOAD_URL = 'https://github.com/ealcobaca/pymfe/releases'


CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: MIT License',
               'Natural Language :: English',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: OS Independent',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']


INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'patsy', 'pandas',
                    'statsmodels', 'texttable']


EXTRAS_REQUIRE = {
    'code-check': [
        'pytest',
        'mypy',
        'liac-arff',
        'flake8',
        'pylint'
    ]
}


setuptools.setup(
    name=NAME,
    version=VERSION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    description=DESCRIPTION,
    download_url=DOWNLOAD_URL,
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
