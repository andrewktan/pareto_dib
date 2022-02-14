import numpy as np

from pareto_dib import pareto_mapper

p = np.loadtxt("data/pxy_test.txt")

pareto, _ = pareto_mapper(p, epsilon=1e-3)
