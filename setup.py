"""Setup for pymfe package."""
import setuptools
import os


my_path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(my_path, 'requirements.txt')) as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()]
    print(install_reqs)

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="pymfe",
    version="0.0.1",
    author="Edesio Alcobaça, Felipe Alves Siqueira, Luis Paulo Faina Garcia",
    author_email="edesio@usp.br, felipe.siqueira@usp.br, lpgarcia@icmc.usp.br",
    description="Meta-feature Extractor",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/ealcobaca/pymfe",
    packages=setuptools.find_packages(),
    install_requires=install_reqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
