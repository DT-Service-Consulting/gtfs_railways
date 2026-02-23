# GTFS Railways

`gtfs-railways` is a Python package designed to analyze and simulate railway networks based on GTFS (General Transit Feed Specification) data stored in SQLite databases. It provides tools for network efficiency simulations, node removal strategies, and visualization of results.

## Features

- Load GTFS data from SQLite databases  
- Compute network efficiency metrics  
- Simulate the impact of node removals (random, targeted, betweenness-based)  
- Visualize efficiency results with customizable plots  
- Utilities for working with railway transit data  

## Installation

Install the package from PyPI using pip:

```bash
pip install gtfs-railways
```

In its current state, the package requires some external packages to be installed manually. 
You can install them using pip as follows:

```bash
pip gtfs_railways/external_packages/osmread
pip gtfs_railways/external_packages/gtfspy
```

## Examples

- **example_01.py**
  - Minimal working example of the P-space function.
    The P-space is the ensemble of all possible paths in the railway network.


- **example_02.py**
  - Minimal working example of the GTC function.
    GTC is the computational cost to the calculation of all connections between two nodes in the P-space.

    The travel cost is the calculation of all possible paths between any two nodes in the P-space.


- **example_03.py**
  - Minimal working example of the efficiency_graph function.
    The efficiency_graph function has been optimized (version 0 to 5).


- **example_04.py**
   - Minimal working example of the nodes removal simulation on a single graph.