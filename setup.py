from setuptools import setup, find_packages

setup(
    name='gtfs_railways',
    version='0.1.4',
    packages=find_packages(include=['gtfs_railways', 'gtfs_railways.*']),
    description='Functions to work on GTFS railways data',
    author='Praneesh Sharma',
    author_email='praneesh.sharma@dtsc.be',
    url='https://github.com/Praneesh-Sharma',
    install_requires=[
        'pandas',
        'numpy',
        'bokeh',
        'networkx',
        'matplotlib',
        'geopy',
        'thefuzz',
        'ipython',
        'ipykernel',
        'jupyter'
    ],
    python_requires='>=3.6',
)
