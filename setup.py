#try:
#    from setuptools import setup
#except ImportError:
#    from distutils.core import setup

from setuptools import setup, find_packages

setup(
    name='chisel',
    version='0.0.1dev',
    author='Chia-Chi Chang',
    author_email='c3h3.tw@gmail.com',
    packages=find_packages(),
    install_requires=["pandas"],
)
