import numpy as np

def path_length(points):
    pts = np.array(points)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

def curvature_stats(points):
    pts = np.array(points)
    kappa = []
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i-1], pts[i], pts[i+1]
        d1 = p1 - p0
        d2 = p2 - p1
        dd = d2 - d1
        num = abs(d1[0]*dd[1] - d1[1]*dd[0])
        den = (np.linalg.norm(d1) ** 3)
        if den > 1e-6:
            kappa.append(num / den)
    if not kappa:
        return 0.0, 0.0
    return float(np.mean(kappa)), float(np.max(kappa))
