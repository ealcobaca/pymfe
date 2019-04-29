"""Setup for pymfe package."""
import setuptools
import os

# get __version__ from _version.py
ver_file = os.path.join('imblearn', '_version.py')
with open(ver_file) as f:
    exec(f.read())


NAME = 'pymfe'


VERSION = __version__


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


INSTALL_REQUIRES = ['numpy', 'scipy', 'sklearn', 'patsy', 'pandas']


EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov',
    ],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
    ]
}


setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="tex/markdown",
    license=LICENSE,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    extras_requires=EXTRAS_REQUIRE,
)
