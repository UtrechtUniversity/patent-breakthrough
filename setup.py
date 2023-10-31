# based on https://github.com/pypa/sampleproject - MIT License
from setuptools import setup, find_packages

setup(
    name='docembedder',
    author='UU Research Engineering Team',
    description='Package for creating document embeddings',
    version="0.1.0",
    long_description='Package for creating document embeddings',
    packages=find_packages(exclude=['data', 'docs', 'tests', 'examples']),
    python_requires='>=3.8',
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
        "polars>=0.16",
        "pyarrow",
        "matplotlib",
        "tqdm",
        "h5py",
        "hyperopt"
    ]
)
