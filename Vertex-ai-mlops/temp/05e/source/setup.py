from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = ['tensorflow_io']

setup(
    name = 'trainer',
    version = '0.1',
    install_requires = REQUIRED_PACKAGES, 
    packages = find_packages(),
    include_package_data = True,
    description='Training Package'
)
