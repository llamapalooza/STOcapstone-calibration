import numpy as np

def sqdist(p, q):
    """
    Returns the squared distance between two 2D points.
    """
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

def area(vertices):
    """
    Returns the area of a simple polygon with vertices as ordered in *vertices*.
    Returns a lower bound for the area if the polygon has intersecting edges.
    """
    return sum([ p[0]*q[1] - q[0]*p[1]
                 for (p, q) in zip(vertices, vertices[1:] + [vertices[0]])
                 ]) / 2

def homogenous(*coords):
    """
    Returns a column matrix containing the coordinates in *coords* followed by a 1.
    """
    x = []
    for x_i in coords:
        x.append([x_i])
    x.append([1])
    return np.matrix(x)

def ctoh(point):
    """
    Returns the homogenous coordinates of a Cartesian point.
    """
    return np.matrix([[point[0]], [point[1]], [1]])

def htoc(point):
    """
    Returns the Cartesian coordinates of a homogenous point.
    """
    return [point[0, 0] / point[2, 0], point[1, 0] / point[2, 0]]
