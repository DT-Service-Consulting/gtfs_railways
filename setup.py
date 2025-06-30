from setuptools import setup, find_packages
import os 

setup(
    name='gtfs_railways',            # Replace with your package name
    version='0.1.4',
    packages=find_packages(),      # Automatically find packages in your folder
    description='Functions to work on GTFS railways data',
    author='Praneesh Sharma',
    author_email='praneesh.sharma@dtsc.be',
    url='https://github.com/Praneesh-Sharma',  # Optional
    install_requires=[
        'pandas',
        'numpy',
        'bokeh',
        'networkx',
        'matplotlib',
        'geopy',
        'thefuzz',
        'gtfspy',
        'ipython'
    ],
    python_requires='>=3.6',
)
