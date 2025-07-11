from setuptools import setup, find_packages
import os

# Read README.md for long_description if it exists
this_directory = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(this_directory, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "A Python library to process and visualize GTFS (General Transit Feed Specification) data for railways."


setup(
    name='gtfs_railways',
    version='0.1.4',
    packages=find_packages(include=['gtfs_railways', 'gtfs_railways.*']),
    description='Functions to work on GTFS railways data',
    long_description_content_type='text/markdown',
    author='Renzo Massobrio',
    author_email='renzo.massobrio@uantwerpen.be',
    maintainer='Marco Di Gennaro, Praneesh Sharma',
    maintainer_email='marco.digennaro@dtsc.be, praneesh.sharma@dtsc.be',
    url='https://github.com/DT-Service-Consulting/gtfs_railways',
    project_urls={
        'Homepage': 'https://github.com/DT-Service-Consulting/gtfs_railways',
    },
    install_requires=[
        'pandas',
        'numpy',
        'bokeh',
        'networkx',
        'matplotlib',
        'geopy',
        'thefuzz',
        'ipython',
        'pytest',
        'pytest-cov'
    ],
    classifiers=[
        'License :: Apache 2.0',
    ],
)
