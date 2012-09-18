

def Point(x, y, z=0):
    """
    Creates a point with coordinates *x*, *y*, and *z*.
    """
    return [x, y, z]

def get_x(p):
    """
    Gets the x coordinate from a point *p*.
    """
    return p[0]

def set_x(p, val):
    """
    Sets the x coordinate of a point *p* to *val*.
    """
    p[0] = val

def get_y(p):
    """
    Gets the y coordinate from a point *p*.
    """
    return p[1]

def set_y(p, val):
    """
    Sets the y coordinate of a point *p* to *val*.
    """
    p[1] = val

def get_z(p):
    """
    Gets the z coordinate from a point *p*.
    """
    return p[2]

def set_z(p, val):
    """
    Sets the z coordinate of a point *p* to *val*.
    """
    p[2] = val

def sqdist(p, q):
    """
    Returns the squared distance between two points (using the x and y dimensions).
    """
    return (get_x(p) - get_x(q))**2 + (get_y(p) - get_y(q))**2

def segments(P):
    """
    Returns a list of 2-tuples, endpoint pairs of segments of a simple polygon
    with vertices as ordered in *P*.
    """
    return zip(P, P[1:] + [P[0]])

def A_2x(P):
    """
    Returns twice the area of a simple polygon with vertices as ordered in *P*.
    (Returns a lower bound for the area if the polygon has intersecting edges.)
    """
    
    return abs(sum([get_x(p)*get_y(q) - get_x(q)*get_y(p) for (p, q) in segments(P)]))
