<div align="center">
<img src="https://github.com/morphomatics/morphomatics.github.io/blob/master/images/logo_cyan.png?raw=true" width="250" alt="Morphomatics"/>
</div>

# Morphomatics: Geometric morphometrics in non-Euclidean shape spaces

Morphomatics is an open-source Python library for (statistical) shape analysis developed within the [geometric data analysis and processing](https://www.zib.de/visual/geometric-data-analysis-and-processing) research group at Zuse Institute Berlin.
It contains prototype implementations of intrinsic manifold-based methods that are highly consistent and avoid the influence of unwanted effects such as bias due to arbitrary choices of coordinates.

Detailed information and tutorials can be found at https://morphomatics.github.io/

## Installation

Morphomatics can be installed directly from github using the following command:
```
pip install git+https://github.com/morphomatics/morphomatics.git#egg=morphomatics
```
For instructions on how to set up `jaxlib`, please refer to the [JAX install guide](https://github.com/google/jax#installation).

## Dependencies
* jax/jaxlib
* jraph
* flax
* optax

Optional
* pymanopt
* sksparse
