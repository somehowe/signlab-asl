"""Assign clips to deterministic, class-covered evaluation splits."""

import numpy as np


SPLIT_NAMES = ("train", "validation", "test")


def split_indices(labels, groups, seed):
    """Return index arrays with a 60/20/20 target and no overlap in group IDs."""
    random = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes, encoded = np.unique(labels, return_inverse=True)
    totals = np.bincount(encoded)
    if np.any(totals < 3):
        raise ValueError("Each sign needs at least three usable clips.")

    if groups is None:
        splits = [[], [], []]
        for label in classes:
            clips = random.permutation(np.flatnonzero(labels == label))
            size = max(1, round(0.2 * len(clips)))
            for split, part in zip(splits, (clips[2 * size :], clips[:size], clips[size : 2 * size])):
                split.extend(part)
        return tuple(np.asarray(sorted(indices)) for indices in splits)

    groups = np.asarray(groups)
    unique, group_ids = np.unique(groups, return_inverse=True)
    by_group = np.zeros((len(unique), len(classes)), dtype=int)
    np.add.at(by_group, (group_ids, encoded), 1)
    if np.any((by_group > 0).sum(axis=0) < 3):
        raise ValueError("Every sign needs examples from at least three groups for a disjoint split.")

    size = max(1, round(0.2 * len(unique)))
    targets = np.array([0.6, 0.2, 0.2])[:, None]
    best = None
    best_cost = float("inf")
    for _ in range(1024):
        shuffled = random.permutation(len(unique))
        parts = (shuffled[2 * size :], shuffled[:size], shuffled[size : 2 * size])
        counts = np.array([by_group[part].sum(axis=0) for part in parts])
        if np.any(counts == 0):
            continue
        cost = float(np.square(counts / totals - targets).sum())
        if cost < best_cost:
            best, best_cost = parts, cost
    if best is None:
        raise ValueError("Could not find class-covered group splits. Use signs with more group overlap.")
    return tuple(np.flatnonzero(np.isin(group_ids, part)) for part in best)
