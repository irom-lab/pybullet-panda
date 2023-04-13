import setuptools


setuptools.setup(
    name='pybullet-panda',
    packages=setuptools.find_packages(),
    install_requires=[
        'pybullet',
        'omegaconf',
        'cloudpickle',
        'dill',
    ],
)
