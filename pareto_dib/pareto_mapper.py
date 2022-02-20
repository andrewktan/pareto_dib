from time import time

import numpy as np

from .classification_utils import (combine_cmaps, entropy, entropy_diff,
                                   merge_joint, mutual_information,
                                   single_cluster_cmap, weighted_jsd)
from .pareto_set import ParetoSet


def pareto_mapper(p, epsilon=1e-8):
    """
    Pareto Mapper

        Parameters:
                p (numpy.ndarray): joint distribution
                eps (float, optional): search depth, default 1e-12

        Returns:
                pset (ParetoSet): DIB Pareto frontier
                run_stats (dict): performance statistics
    """
    n = p.shape[0]

    # plot pruned search points
    cmap0 = {x: {x, } for x in range(p.shape[0])}

    pset = ParetoSet()
    Imax = mutual_information(p)
    Hmax = -entropy(p.sum(1))
    new_point = (Hmax, Imax, cmap0, p)

    Q = []
    Q.append(new_point)

    tried = set()
    count = 0
    t0 = time()

    while Q:
        count += 1
        point = Q.pop()
        nent, mi, cmap, pc = point

        for j in range(len(cmap)):
            for k in range(len(cmap)):
                if j >= k:
                    continue

                dy = weighted_jsd(pc[j, :], pc[k, :])
                dx = entropy_diff(pc.sum(1)[[j, k]])

                dcmap = single_cluster_cmap(len(cmap), (j, k))

                ncmap = combine_cmaps(cmap, dcmap)

                new_point = (nent + dx,
                             mi - dy,
                             ncmap,
                             merge_joint(pc, dcmap))

                # evaluate point, and attempt insertion into Pareto Set
                pset.add(new_point)
                a = pset.is_pareto(new_point)
                b = np.exp(-pset.distance(new_point) /
                           epsilon) > np.random.rand()

                # check that point has not been tried
                pid = (round(new_point[0], 8), round(new_point[1], 8))
                if (a or b) and (pid not in tried):
                    tried.add(pid)
                    Q.append(new_point)

    tf = time()

    # save run stats
    run_stats = {}
    run_stats['n'] = n
    run_stats['searched'] = count
    run_stats['time'] = float(tf - t0)
    run_stats['pareto_size'] = len(pset)
    run_stats['epsilon'] = epsilon

    return pset, run_stats
