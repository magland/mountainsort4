from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    scripts=[],
    install_requires=[
        'dask',
        'pybind11',
        'isosplit5==0.1.3',
        'numpy',
        'h5py',
        'sklearn',
        'spikeinterface>=0.90'
    ]
)