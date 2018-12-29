# -*- conding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyMFE",
    version="0.0.1",
    author="Edesio Alcobaça, Felipe Alves Siqueira",
    author_email="[email?], felipe.siqueira@usp.br",
    description="Metafeature extractor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/.../pyMFE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
