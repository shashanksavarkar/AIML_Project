"""Turns one wafer map into a row of numerical features.

Pixel values in a WM-811K wafer map:
    0 = off-wafer (no die position here)
    1 = good die
    2 = defective die

Every feature is scaled so it does not depend on the grid size.
Labels (failureType) are never used here: they are only the answer.
"""
import numpy as np
from scipy import ndimage  # used to group touching defect pixels into clusters

FEATURE_NAMES = [
    "n_dies", "n_defects", "defect_density",
    "center_density", "mid_density", "edge_density",
    "left_frac", "top_frac",
    "x_spread", "y_spread", "mean_radius",
    "max_row_density", "max_col_density",
    "n_clusters", "largest_cluster_frac",
]


def extract_features(wafer_map):
    """Return a dict with one value per name in FEATURE_NAMES, for one wafer map."""
    a = np.asarray(wafer_map)                       # the wafer map as a 2D numpy grid
    valid = a > 0                                   # True where a die exists (good or defective)
    defect = a == 2                                 # True where a defective die exists
    n_dies = int(valid.sum())                       # total die positions on the wafer
    n_defects = int(defect.sum())                   # total defective die positions

    if n_dies == 0:                                 # nothing to measure on an empty map
        return dict.fromkeys(FEATURE_NAMES, 0.0)

    rows, cols = np.indices(a.shape)                # row number and column number of every pixel
    cy = rows[valid].mean()                         # centre of the die area, row direction
    cx = cols[valid].mean()                         # centre of the die area, column direction
    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)   # distance of every pixel from that centre
    rn = dist / (dist[valid].max() + 1e-9)          # normalised distance: 0 = centre, 1 = wafer edge

    def density(region):                            # defective dies / all dies inside a region
        dies = (valid & region).sum()
        return float((defect & region).sum() / dies) if dies else 0.0

    center = rn < 0.33                              # inner third of the wafer
    mid = (rn >= 0.33) & (rn < 0.66)                # middle third
    edge = rn >= 0.66                               # outer third

    # Share of defects on the left half and on the top half (0.5 = balanced)
    left_frac = float((defect & (cols < cx)).sum() / max(n_defects, 1))
    top_frac = float((defect & (rows < cy)).sum() / max(n_defects, 1))

    if n_defects:                                   # spread and position of defects, scaled by map size
        x_spread = float(cols[defect].std() / a.shape[1])
        y_spread = float(rows[defect].std() / a.shape[0])
        mean_radius = float(rn[defect].mean())
    else:
        x_spread = y_spread = mean_radius = 0.0

    # Worst row and worst column: a scratch or a line shows up as one row or column with many defects
    row_valid = valid.sum(axis=1)
    col_valid = valid.sum(axis=0)
    row_rate = np.divide(defect.sum(axis=1), row_valid, out=np.zeros(len(row_valid)), where=row_valid > 0)
    col_rate = np.divide(defect.sum(axis=0), col_valid, out=np.zeros(len(col_valid)), where=col_valid > 0)

    # Clusters: groups of defect pixels that touch each other (a blob, a scratch, a ring piece...)
    labels, n_clusters = ndimage.label(defect)      # label each cluster 1, 2, 3...; background stays 0
    if n_clusters:
        sizes = np.bincount(labels.ravel())[1:]     # size of each cluster (index 0 is the background, skipped)
        largest_cluster_frac = float(sizes.max() / n_defects)  # share of defects in the biggest cluster
    else:
        largest_cluster_frac = 0.0

    return {
        "n_dies": float(n_dies),
        "n_defects": float(n_defects),
        "defect_density": n_defects / n_dies,
        "center_density": density(center),
        "mid_density": density(mid),
        "edge_density": density(edge),
        "left_frac": left_frac,
        "top_frac": top_frac,
        "x_spread": x_spread,
        "y_spread": y_spread,
        "mean_radius": mean_radius,
        "max_row_density": float(row_rate.max()),
        "max_col_density": float(col_rate.max()),
        "n_clusters": float(n_clusters),
        "largest_cluster_frac": largest_cluster_frac,
    }


if __name__ == "__main__":                          # self-check: runs only when this file is run directly
    test = np.array([[0, 1, 1],
                     [1, 2, 1],
                     [0, 1, 0]])                    # 6 die positions, 1 defect in the middle
    f = extract_features(test)
    assert f["n_dies"] == 6 and f["n_defects"] == 1
    assert abs(f["defect_density"] - 1 / 6) < 1e-9
    assert f["n_clusters"] == 1
    assert set(f) == set(FEATURE_NAMES)
    print("features.py self-check passed:", f)
