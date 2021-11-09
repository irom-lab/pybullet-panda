import setuptools

setuptools.setup(
    name='panda',
    version='1.0',
    description='Custom Franka Panda Environment in PyBullet by IRoM Lab',
    author='Allen Z. Ren',
    author_email='allen.ren@princeton.edu',
    packages=setuptools.find_packages(),
    install_requires=['pybullet'],
)
