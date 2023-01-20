# based on https://github.com/pypa/sampleproject - MIT License
from setuptools import setup, find_packages

import versioneer

setup(
    name='docembedder',
    author='UU Research Engineering Team',
    description='Package for creating document embeddings',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description='Package for creating document embeddings',
    packages=find_packages(exclude=['data', 'docs', 'tests', 'examples']),
    python_requires='~=3.6',
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "pandas",
        "dill",
        "nltk",
        "gensim",
        "bpemb",
        "polars",
        "pyarrow",
        "matplotlib",
        "tqdm",
        "h5py",
    ]
)
