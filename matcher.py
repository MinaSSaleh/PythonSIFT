def fast_keypoint_matching(desc1, desc2, ratio_threshold=0.75):
    """Lowe ratio test for robust keypoint matching."""
    import numpy as np
    distances = np.linalg.norm(desc1[:, None] - desc2[None, :], axis=2)
    sorted_idx = np.argsort(distances, axis=1)
    matches = []
    for i, (best, second) in enumerate(sorted_idx[:, :2]):
        if distances[i, best] < ratio_threshold * distances[i, second]:
            matches.append((i, best, distances[i, best]))
    return matches
