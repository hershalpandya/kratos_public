# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    kratos_readme = f.read()

with open('LICENSE') as f:
    kratos_license = f.read()

requirements = ['numpy', 'scipy', 'matplotlib', 'pandas', 'radiotools', 'astropy', 'h5py',
                'flake8', 'pytest']

setup(
    name='kratos',
    version='0.0.2',
    description='Kosmic-ray Radio Analysis TOolS',
    long_description=kratos_readme,
    author='',
    author_email='',
    url='https://gitlab.science.ru.nl/hpandya/kratos',
    license=kratos_license,
    packages=find_packages(exclude=('docs', 'testing', 'data')),
    setup_requires=['wheel'],
    install_requires=requirements
    )
