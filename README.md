# Primal Deterministic Information Bottleneck

![python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![tags](https://img.shields.io/github/v/tag/andrewktan/pareto_dib)
![tests](https://github.com/andrewktan/pareto_dib/actions/workflows/ci.yml/badge.svg)
![license](https://img.shields.io/github/license/andrewktan/pareto_dib)

This code is an implementation of the Pareto Mapper and Symmetric Pareto Mapper algorithms presented in "Pareto-optimal clustering with the primal deterministic information bottleneck."

## Abstract
At the heart of both lossy compression and clustering is a trade-off between the fidelity of the learned representation and its size.
We focus on the Deterministic Information Bottleneck (DIB) formulation of lossy compression, which can be readily interpreted as a clustering problem, owing to the fact that its search space is that of hard clusterings.
Our goal is to motivate the task of mapping out the Pareto frontier of these two-objective trade-offs in full generality.
To this end, we introduce the primal DIB problem, which we argue results in a much richer frontier than its previously studied dual counterpart.
We present an algorithm for mapping out the Pareto frontier of the DIB trade-off that is also applicable to most two-objective clustering problems.
We study general properties of the Pareto set, and give both analytic and numerical evidence for the logarithmic sparsity of the frontier in general.
We present evidence for the polynomial scaling of our algorithm despite the super-exponential search space;
and additionally propose a modification to the algorithm that can be used where sampling noise is expected to be significant.
Finally, we use our algorithm to map the DIB frontier of three different tasks: compressing the English alphabet, extracting informative color classes from natural images, and compressing a group theory inspired dataset, revealing interesting features of frontier, and demonstrating how the structure of the frontier can be used for model selection with a focus on points previously hidden by the cloak of the convex hull.

## Installation

A stable release can be found on the Python Package Index (PyPI) and can be installed with `pip install pareto_dib`.

## Usage

The main methods in the package are as follows:
- Pareto Mapper: `from pareto_dib import pareto_mapper`
- Symmetric Pareto Mapper: `from pareto_dib import symmetric_pareto_mapper`
- Plotting utility: `from pareto_dib import pareto_plot`

### Pareto Mapper

An example use case of Pareto Mapper is provided below.
The `pareto_mapper` function takes a normalized joint distribution of variables (X, Y), `pxy` (numpy.ndarray) with shape `(|X|, |Y|)`.
The search depth is `epsilon=1e-8`.

```
pset, _ = pareto_mapper(pxy, epsilon=1e-8)
ax = pareto_plot(pset)
```

### Symmetric Pareto Mapper

An example use case of Symmetric Pareto Mapper is provided below.
The `symmetric_pareto_mapper` function takes a normalized joint distribution of variables (X_1, X_2, Y), `pxxy` (numpy.ndarray) with shape `(|X|, |X|, |Y|)`.
The search depth is `epsilon=1e-8`.

```
pset, _ = pareto_mapper(pxxy, epsilon=1e-8)
ax = pareto_plot(pset)
```

## Examples

The datasets presented in "Pareto-optimal clustering with the primal deterministic information bottleneck" are provided in the `examples/data' directory.

For details on the creation of the datasets and a discussion of the frontier, please refer to the paper.

### English alphabet
To reproduce this plot, run `python3 examples/example.py --dataset alpha27`.

![English Alphabet frontier](https://github.com/andrewktan/pareto_dib/blob/main/images/alpha27.jpg)

### Colors
To reproduce this plot, run `python3 examples/example.py --dataset colors`.

![Colors frontier](https://github.com/andrewktan/pareto_dib/blob/main/images/colors.jpg)

### Group datasets
To reproduce this plot, run `python3 examples/example.py --dataset Z40x`.

![Z40x group frontier](https://github.com/andrewktan/pareto_dib/blob/main/images/Z40x.jpg)

To reproduce this plot, run `python3 examples/example.py --dataset pauli`.

![Pauli group frontier](https://github.com/andrewktan/pareto_dib/blob/main/images/pauli.jpg)

## Version History

* 0.0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Citation
