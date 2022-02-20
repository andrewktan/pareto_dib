import matplotlib.pyplot as plt
import numpy as np

from pareto_dib import pareto_mapper


def pareto_plot(pset, scale='standard'):
    """
    Plot ParetoSet with boundary lines and convex hull.

        Parameters:
                pset (ParetoSet): Pareto set
                scale: 'standard' or 'symmetric' for ParetoMapper and
                    SymmetricParetoMapper output respectively
    """
    fig = plt.figure()
    ax = fig.gca()

    # scale
    if scale == 'standard':
        sval = 1
    elif scale == 'symmetric':
        sval = 2
    else:
        raise f"scale: {scale} not supported."

    # plot frontier
    plist = pset.to_list()
    points = np.array([(x[0] / sval, x[1]) for x in plist])
    ax.plot(points[:, 0],
            points[:, 1],
            color='k',
            marker='.',
            markersize=6,
            alpha=1.,
            linewidth=0,
            zorder=3)

    # plot boundary lines
    xr1 = np.linspace(points[:, 0].min(), -points[:, 1].max())
    xr2 = np.linspace(-points[:, 1].max(), 0.)
    xr = np.r_[xr1, xr2]

    ax.plot(xr1,
            np.ones(xr1.shape) * points[:, 1].max(),
            'k-',
            linewidth=2,
            zorder=2)
    ax.plot(xr2, -xr2, 'k-', linewidth=2, zorder=2)

    ax.fill_between(xr,
                    np.ones(xr.shape) * 10,
                    -xr,
                    color='pink',
                    zorder=1)
    ax.fill_between(xr,
                    np.ones(xr.shape) * 10,
                    np.ones(xr.shape) * points[:, 1].max(),
                    color='pink',
                    zorder=1)

    # plot hull
    hull_idx = [0, ]
    while hull_idx[-1] < points.shape[0] - 1:
        curr_idx = hull_idx[-1]
        # find max slope
        max_slope = None
        msidx = None
        for idx in range(curr_idx + 1, points.shape[0]):
            tslope = (points[curr_idx, 1] - points[idx, 1]) / \
                (points[curr_idx, 0] - points[idx, 0])
            if max_slope is None or tslope > max_slope:
                max_slope = tslope
                msidx = idx

        hull_idx.append(msidx)

    ax.fill_between(points[hull_idx, 0],
                    np.zeros(points[hull_idx, 0].shape),
                    points[hull_idx, 1],
                    color='grey',
                    alpha=0.2,
                    zorder=0)

    # set axis limits
    ax.set_title("DIB Frontier")
    ax.set_xlabel("-H(Z) / 2 [bits]" if sval == 2 else "-H(Z) [bits]")
    ax.set_ylabel("I(Z; Y) [bits]")
    ax.set_xlim(np.min(points[:, 0]), np.max(points[:, 0]))
    ax.set_ylim(0., points[:, 1].max() * 1.05)
    ax.xaxis.set_major_formatter(lambda x, _: f"{np.abs(x):.1f}")

    return ax


if __name__ == "__main__":
    pxy = np.load(
        "/Users/andrew/Documents/phd/PAPER-pareto/code/release/examples/data/pxy_alpha27.npy")

    pset, _ = pareto_mapper(pxy, epsilon=1e-12)

    pareto_plot(pset)

    plt.plot()
