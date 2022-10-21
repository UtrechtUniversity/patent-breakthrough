# based on https://github.com/pypa/sampleproject - MIT License

from setuptools import setup, find_packages

setup(
    name='docembedder',
    version='0.0.1',
    author='UU Research Engineering Team',
    description='Package for creating document embeddings',
    long_description='Package for creating document embeddings',
    packages=find_packages(exclude=['data', 'docs', 'tests', 'examples']),
    python_requires='~=3.6',
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "torch",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "pandas",
        "dill",
        "nltk",
        "gensim",
        "bpemb",
        "pytest"
    ]
)
