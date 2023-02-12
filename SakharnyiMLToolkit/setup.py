from setuptools import setup, find_packages

setup(
    name='SakharnyiMLToolkit',
    version='0.23',
    author='Siarhei Sakharnyi',
    description='Combined package for feature extraction, hyperparameter tuning, and validation schema',
    long_description='A package that provides combined solutions for feature extraction, hyperparameter tuning, and validation schema, with a focus on making the process more efficient and easier.',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn', 'hyperopt','pandas', 'fuzzywuzzy']
)

