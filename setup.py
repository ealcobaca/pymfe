"""Setup for pymfe package."""
import setuptools
import os
import pymfe


NAME = 'pymfe'


VERSION = pymfe.__version__


AUTHOR = 'Edesio Alcobaça, Felipe Alves Siqueira, Luis Paulo Faina Garcia'


AUTHOR_EMAIL = 'edesio@usp.br, felipe.siqueira@usp.br, lpgarcia@icmc.usp.br'


DESCRIPTION = 'Meta-feature Extractor'


with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()


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


INSTALL_REQUIRES = ['numpy', 'scipy', 'sklearn', 'patsy', 'pandas',
                    'statsmodels']


EXTRAS_REQUIRE = {
    'code-check': [
        'pytest',
        'mypy',
        'liac-arff',
        'flake8',
        'pylint'
    ],
    'tests': [
        'pytest',
        'pytest-cov',
        'liac-arff'
    ],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'liac-arff'
    ]
}


setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
