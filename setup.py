from setuptools import setup, find_packages

setup(
   name='mixtures',
   version='0.0.2',
   description='A python package for exploring univariate and multivariate gaussian mixture models',
   author='Joseph Blazick',
   author_email='joe.blazick@yahoo.com',
   package_dir={"": "src"},
   packages=find_packages(where="src"),   
   install_requires=['distcan', 'scipy', 'numpy', 'pandas', 'matplotlib'], #external packages as dependencies
)