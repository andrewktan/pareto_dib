import argparse

import matplotlib.pyplot as plt
import numpy as np

from pareto_dib import pareto_mapper, pareto_plot, symmetric_pareto_mapper

# Supported datasets

DSETS = ['alpha27', 'colors', 'Z40x', 'pauli']
dsets_str = ", ".join(dset for dset in DSETS)

# Parse arguments

parser = argparse.ArgumentParser(description="Pareto Mapper examples")
parser.add_argument('--dataset',
                    type=str,
                    default='Z40x',
                    help=f"Dataset. Choose from [{dsets_str}].")

args = parser.parse_args()
dset = args.dataset

# Run Pareto Mapper

if dset == 'alpha27':
    pxy = np.load("data/pxy_alpha27.npy")
    pset, _ = pareto_mapper(pxy, epsilon=1e-8)
    ax = pareto_plot(pset)
elif dset == 'colors':
    pxy = np.load("data/pxy_colors.npy")
    pset, _ = pareto_mapper(pxy, epsilon=1e-8)
    ax = pareto_plot(pset)
elif dset == 'Z40x':
    pxy = np.load("data/pxy_Z40x.npy")
    pset, _ = symmetric_pareto_mapper(pxy, epsilon=1e-8)
    ax = pareto_plot(pset, scale='symmetric')
elif dset == 'pauli':
    pxy = np.load("data/pxy_pauli.npy")
    pset, _ = symmetric_pareto_mapper(pxy, epsilon=1e-8)
    ax = pareto_plot(pset, scale='symmetric')
else:
    raise Exception(f"Dataset: '{dset}' not found.")

ax.set_title(f"DIB Frontier ({dset})")
plt.show()
