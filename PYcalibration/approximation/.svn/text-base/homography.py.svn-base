import numpy as np
import numpy.linalg

from coord import sqdist, homogenous, ctoh, htoc

def homography(FROM, TO):
    """
    Returns the homography determined by four points *FROM* and their images *TO*.
    Computed using SVD decomposition as in ``A Flexible New Technique for Camera
    Calibration'' (Zhang).
    """
    assert len(FROM) == 4
    assert len(TO) == 4
    
    # homography
    #   H = [[ h_1, h_2, h_3 ],
    #        [ h_4, h_5, h_6 ],
    #        [ h_7, h_8, h_9 ]]
    #
    # pre-image/image pairs
    #   \bf{x}_i = [x_i, y_i, z_i]^T \to \bf{x}_i' = [u_i, v_i, w_i]^T
    #
    # we set
    #   H\bf{x}_i \times \bf{x}_i' = 0
    #
    # ...
    
    A = []
    
    def A_i(FROM, TO):
        x = FROM[0, 0]
        y = FROM[1, 0]
        z = FROM[2, 0]
        
        u = TO[0, 0]
        v = TO[1, 0]
        w = TO[2, 0]
        
        return [ [    0,    0,    0, -w*x, -w*y, -w*z,  v*x,  v*y,  v*z ],
                 [  w*x,  w*y,  w*z,    0,    0,    0, -u*x, -u*y, -u*z ],
                 [ -v*x, -v*y, -v*z,  u*x,  u*y,  u*z,    0,    0,    0 ] ]
    
    for i in range(len(FROM)):
        A.extend(A_i(FROM[i], TO[i]))
    
    # SVD decomposition of A (note that V_T means V*)
    U, S, V_T = np.linalg.svd(np.matrix(A))
    
    # bottom row of V* is rightmost column of V (which is in the null space of A)
    H_l = V_T[-1]
    
    # reconstruct homography H
    return np.reshape(H_l, (3, 3))

def best_into_mapping(domain, H, codomain):
    """
    Associates each point in *domain* with the point in *codomain* nearest its image
    under homography *H*. Returns an ordered list of images corresponding to points
    in *domain*, along with some indication of how good or bad the match is.
    """
    RANGE = []
    total_error = 0
    
    for preimage in domain:
        modeled = H * preimage
        
        min_error = None
        image = None
        for point in codomain:
            error = sqdist(htoc(modeled), htoc(point))
            if min_error == None or error < min_error:
                min_error = error
                image = point
        
        total_error += min_error
        RANGE.append(image)
    
    return RANGE, total_error

