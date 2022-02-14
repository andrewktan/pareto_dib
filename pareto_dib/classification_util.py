from collections import defaultdict

import numpy as np
from scipy.special import rel_entr


def kl(p, q):
    if np.sum(p) > 0 and np.sum(q) > 0:
        p = p / np.sum(p)
        q = q / np.sum(q)
    else:
        return 100000.

    return np.sum(rel_entr(p, q)) / np.log(2)


def jsd(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)

    return (kl(p, (p + q) / 2) + kl(q, (p + q) / 2)) / 2


def entropy(p):
    return - np.sum(p * np.log2(p, out=np.zeros_like(p), where=p != 0))


def mutual_information(pxy):
    px = np.sum(pxy, 1).reshape(-1, 1)
    py = np.sum(pxy, 0)
    pxpy = px * py

    return kl(pxy[:], pxpy[:])


def entropy_diff(px):
    pt = px.sum()

    return -np.sum(
        px * np.log2(
            np.divide(px, pt, out=np.zeros_like(px), where=pt > 0),
            out=np.zeros_like(px), where=px != 0)
    )


def weighted_jsd(*argv):
    """takes n unnormalized slices of the joint matrix"""

    ps = np.zeros((len(argv), argv[0].size))
    w = np.zeros(len(argv))
    m = np.zeros_like(argv[0])

    for i, p in enumerate(argv):
        ps[i, :] = p[:]
        w[i] = np.sum(p)
        m += p

    m /= np.sum(w)

    wjsd = 0.

    for i in range(len(argv)):
        wjsd += kl(ps[i, :], m) * w[i]

    return wjsd


def merge_joint(pxy, cmap):
    ret = np.zeros((len(cmap), pxy.shape[1]))
    for c, S in cmap.items():
        for y in S:
            ret[c, :] += pxy[y, :]

    return ret


def single_cluster_cmap(n, cluster):
    c = 0
    cmap = defaultdict(set)

    for k in range(n):
        if k not in cluster:
            cmap[c].add(k)
            c += 1
    for k in cluster:
        cmap[c].add(k)

    return cmap


def combine_cmaps(cmap1, cmap2):
    kmax = 0
    for k in cmap2:
        for elem in cmap2[k]:
            if elem > kmax:
                kmax = elem

    if kmax + 1 != len(cmap1):
        raise Exception("cmaps don't match")

    # apply cmap1 first then cmap2
    cmap = defaultdict(set)

    for c, cluster in enumerate(cmap2):
        for k in cmap2[cluster]:
            for elem in cmap1[k]:
                cmap[c].add(elem)

    return cmap
