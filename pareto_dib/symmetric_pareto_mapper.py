from time import time

import numpy as np

from .classification_utils import (combine_cmaps, entropy, merge_joint_sym,
                                   mutual_information, single_cluster_cmap)
from .pareto_set import ParetoSet


def symmetric_pareto_mapper(p3d, epsilon=1e-12):
    """
    Symmetric Pareto Mapper

        Parameters:
                p (numpy.ndarray): joint distribution p_{X1 X2; Y}
                eps (float, optional): search depth, default 1e-12

        Returns:
                pset (ParetoSet): DIB pset frontier
                run_stats (dict): performance statistics
    """
    # params
    N = p3d.shape[0]

    # plot pruned search points
    Zdim = p3d.shape[2]
    cmap0 = {x: {x, } for x in range(p3d.shape[0])}

    pset = ParetoSet()
    Imax = mutual_information(p3d.reshape(-1, Zdim))
    Hmax = -entropy(p3d.sum(-1).reshape(-1))
    new_point = (Hmax, Imax, cmap0, p3d)

    Q = []
    Q.append(new_point)

    tried = set()
    count = 0
    t0 = time()

    while Q:
        count += 1
        point = Q.pop()
        nent, mi, cmap, pc3d = point

        for j in range(len(cmap)):
            for k in range(len(cmap)):
                if j >= k:
                    continue

                dcmap = single_cluster_cmap(len(cmap), (j, k))
                ncmap = combine_cmaps(cmap, dcmap)

                p_new3d = merge_joint_sym(pc3d, dcmap)

                H = entropy(p_new3d.sum(-1).reshape(-1))
                I = mutual_information(p_new3d.reshape(-1, Zdim))

                new_point = (-H,
                             I,
                             ncmap,
                             p_new3d)

                # evaluate point, and attempt insertion into Pareto Set
                pset.add(new_point)
                a = pset.add(point)
                b = np.exp(-pset.distance(point) / epsilon) > np.random.rand()

                # check that point has not been tried
                pid = (round(new_point[0], 8), round(new_point[1], 8))
                if (a or b) and (pid not in tried):
                    tried.add(pid)
                    Q.append(new_point)

    tf = time()

    # save run stats
    run_stats = {}
    run_stats['N'] = N
    run_stats['searched'] = count
    run_stats['time'] = float(tf - t0)
    run_stats['pset_size'] = len(pset)
    run_stats['epsilon'] = epsilon

    # save
    return pset, run_stats
