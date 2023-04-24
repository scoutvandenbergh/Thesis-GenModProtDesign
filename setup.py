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
    description="This GitHub project involves designing new protein sequences through the use of cutting-edge, Variational AutoEncoder-based models (VAE), generative models. Additionally, it introduces a novel evaluation metric called the Fréchet ESM Distance (FED) to evaluate these newly designed protein sequences.",
    author="Scout Van den Bergh",
    author_email="scoutvandenbergh@gmail.com",
    url="https://github.com/scoutvandenbergh/Thesis-GenModProtDesign",
    install_requires=[
        "numpy",
        "torch",
        "pytorch-lightning",
        "h5torch",
        "biopython",
        "rotary_embedding_torch",
        "fair-esm"
    ],
)
