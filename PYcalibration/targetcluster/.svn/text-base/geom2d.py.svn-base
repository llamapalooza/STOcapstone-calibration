import numpy as np
from numpy.linalg import svd

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
    return sum([get_x(p)*get_y(q) - get_x(q)*get_y(p) for (p, q) in segments(P)])

def to_homogenous(p):
    x = get_x(p)
    y = get_y(p)
    return np.matrix([[x], [y], [1]])

def from_homogenous(p):
    return Point(p[0, 0] / p[2, 0], p[1, 0] / p[2, 0])

def homography(preimages, images):
    assert len(preimages) == 4
    assert len(images) == 4
    
    # homography
    #   H = [[ h_1, h_2, h_3 ],
    #        [ h_4, h_5, h_6 ],
    #        [ h_7, h_8, h_9 ]]
    #
    # pre-image/image pairs
    #   \bf{x}_i = [x_i, y_i, z_i]^T \to \bf{x}_i' = [u_i, v_i, w_i]^T
    #
    # we have
    #   H\bf{x}_i \times \bf{x}_i' = 0
    #
    # ...
    
    A = []
    
    def A_1(x, y, z, u, v, w):
        return [    0,    0,    0, -w*x, -w*y, -w*z,  v*x,  v*y,  v*z ]
    
    def A_2(x, y, z, u, v, w):
        return [  w*x,  w*y,  w*z,    0,    0,    0, -u*x, -u*y, -u*z ]
    
    def A_3(x, y, z, u, v, w):
        return [ -v*x, -v*y, -v*z,  u*x,  u*y,  u*z,    0,    0,    0 ]
    
    for i in range(len(preimages)):
        x = get_x(preimages[i])
        y = get_y(preimages[i])
        
        u = get_x(images[i])
        v = get_y(images[i])
        
        A.append(A_1(x, y, 1, u, v, 1))
        A.append(A_2(x, y, 1, u, v, 1))
        A.append(A_3(x, y, 1, u, v, 1))
    
    A = np.matrix(A)
    
    # svd decomposition of A (note that V is actually V*)
    U, S, V = svd(A)
    
    # bottom row of V* is rightmost column of V (which is in the null space of A)
    H_l = V[-1]
    
    # reconstruct homography H
    return np.matrix([ [ H_l[0, 0], H_l[0, 1], H_l[0, 2] ],
                       [ H_l[0, 3], H_l[0, 4], H_l[0, 5] ],
                       [ H_l[0, 6], H_l[0, 7], H_l[0, 8] ] ])

def best_case_mapping(preimages, H, samples):
    E = 0
    images = []
    
    for preimage in preimages:
        prediction = from_homogenous(H * to_homogenous(preimage))
        
        e = None
        image = None
        for sample in samples:
            d = sqdist(prediction, sample)
            if e == None or d < e:
                e = d
                image = sample
        
        E += e
        images.append(image)
    
    return E, images

