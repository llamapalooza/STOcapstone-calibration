import numpy as np
from math import sqrt
from copy import deepcopy

from coord import sqdist, area

def nP4(SET):
    """
    Returns all permutations of four elements from *SET*.
    """
    return [ [a, b, c, d]
             for a in SET
             for b in SET
             if b != a
             for c in SET
             if c != a and c != b
             for d in SET
             if d != a and d != b and d != c ]

def high_pass_filter(points3, threshold):
    """
    Returns a list of points from *points3* who have intensity greater than the
    product of *threshold* and the maximum intensity in *points3*.
    """
    cutoff = threshold * max([ p[2] for p in points3 ])
    return [ p for p in points3 if p[2] > cutoff ]

def _nearest(p, points2):
    """
    Returns the distance from *p* to the nearest other point in *points2*.
    """
    return sqrt(min([ sqdist(p, q) for q in points2 if q != p ]))

def geographic_inliers(points2, factor):
    """
    Returns a list of points from *points2* who have nearest neighbor at distance
    within a factor of two from the arithmetic mean of all nearest neighbor distances.
    """
    avg_nearest = np.average([ _nearest(p, points2) for p in points2 ])
    return [ p
             for p in points2
             if avg_nearest / factor < _nearest(p, points2) < avg_nearest * factor ]

def local_maxima(points3, radius):
    """
    Returns a list of points from *points3* who have no neighbors of greater intensity
    within *radius* pixels. Ties are arbitrarily broken.
    """
    maxima = []
    for p in points3:
        neighbor_zs = [ q[2] for q in points3 if q != p and sqdist(p, q) < radius**2 ]
        
        if neighbor_zs == [] or p[2] > max(neighbor_zs):
            maxima.append(p)
        elif p[2] == max(neighbor_zs):
            # break ties
            p[2] = 0
    
    return maxima

def corners(points2):
    """
    Returns an ordered list containing the four points in *points* which encompass the
    largest area.
    """
    return max([ quad
                 for quad in nP4(points2)
                 if quad[0][0] + quad[0][1] == min([ vertex[0] + vertex[1]
                                                     for vertex in quad ])
                 ], key=area)

def _label(point, centers):
    """
    Returns an index such that *centers*[index] is the point in *centers* nearest *point*.
    """
    min_sqdist = None
    best_label = None
    for c in range(len(centers)):
        if min_sqdist == None or sqdist(point, centers[c]) < min_sqdist:
            min_sqdist = sqdist(point, centers[c])
            best_label = c
    return best_label

def kmeans(points3, centers, epsilon):
    """
    Returns a list of lists of elements of *points3*, k-means clustered in two dimensions
    around the k centers in *centers* with points weighted by third dimension coordinate.
    Iteration until no center moves more than *epsilon*.
    """
    prevcenters = None
    while prevcenters == None or max([ sqdist(center, prevcenter)
                                       for (center, prevcenter) in zip(centers, prevcenters)
                                       ]) > epsilon**2:
        # empty clusters
        clusters = []
        for center in centers:
            clusters.append([])
        
        # recluster points
        for point in points3:
            clusters[_label(point, centers)].append(point)
        
        # find new centers
        prevcenters = deepcopy(centers)
        for i in range(len(centers)):
            centers[i] = np.average(clusters[i],
                                    axis=0,
                                    weights=[ point[2] for point in clusters[i] ])
    
    return clusters

