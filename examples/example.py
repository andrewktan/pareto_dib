import matplotlib.pyplot as plt
import numpy as np

from pareto_dib import pareto_mapper, pareto_plot

pxy = np.load("data/pxy_01.npy")
pset, _ = pareto_mapper(pxy, epsilon=1e-12)
pareto_plot(pset)
plt.show()
