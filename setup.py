import sys
from setuptools import setup, find_packages

sys.path[0:0] = ["evalpgm"]
from version import __version__

setup(
    name="evalpgm",
    python_requires=">3.9.0",
    packages=find_packages(),
    version=__version__,
    license="MIT",
    description="project template",
    author="Gaetan De Waele",
    author_email="gaetan.dewaele@ugent.be",
    url="https://github.com/gdewael/project",
    install_requires=[
        "numpy",
        "torch",
        "pytorch-lightning",
        "h5torch",
        "biopython"
    ],
)
