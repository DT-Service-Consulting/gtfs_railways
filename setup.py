from setuptools import setup, find_packages

setup(
    name='gtfs_railways',
    version='0.1.4',
    packages=find_packages(include=['gtfs_railways', 'gtfs_railways.*']),
    description='Functions to work on GTFS railways data',
    long_description='A Python library to process and visualize GTFS (General Transit Feed Specification) data for railways.',
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
    ],
    python_requires='>=3.6, <3.9',
    classifiers=[
        'License :: Apache 2.0',
    ],
)
