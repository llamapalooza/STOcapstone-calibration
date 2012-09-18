import numpy as np
import math
from numpy.dual import svd, inv, norm
from simulation.projection import CameraParameters, RigidTransformation

def estimate_camera(homographies):
    params = estimate_camera_params(homographies)
    poses = estimate_poses(params, homographies)
    return params, poses

def estimate_camera_params(homographies):
    # Stack the constraints from each homography into a design matrix
    n = len(homographies)
    design_matrix = np.empty((2*n, 6))
    for i, homography in enumerate(homographies):
        design_matrix[2*i]   = homography_vector(homography, 0, 1)
        design_matrix[2*i+1] = homography_vector(homography, 0, 0) \
                             - homography_vector(homography, 1, 1)
    
    # Extract the last right singular vector (ordered by decreasing singular value) and
    # unflatten it to get the image of the absolute conic
    _, _, V = svd(design_matrix)
    B = unflatten_symmetric(V[-1])
    
    return extract_camera_params(B)

def factor_target_pose(world_homography, target_homography):
    """
    Find the rigid transformation that maps target coordinates to world coordinates,
    given homographies that map world coordinates and target coordinates to a common
    coordinate system (i.e., the camera).
    """
    return extract_transformation(np.dot(inv(world_homography), target_homography))

def factor_camera_pose(stereo_transform, camera_transform):
    """
    Find the rigid transformation that maps stereo coordinates to camera coordinates,
    given transformations that map a common coordinate system (i.e., the world) to stereo
    and camera coordinates.
    """
    pose = np.dot(camera_transform, stereo_transform.inverse())
    return RigidTransformation(matrix=pose)

def estimate_poses(params, homographies):
    K_inv = inv(params)
    return [extract_transformation(np.dot(K_inv, H)) for H in homographies]

def extract_transformation(homography):
    """
    Take a 3x3 homography representing a truncated rigid transformation (where *z* is
    assumed to be zero so the third column of the rotation matrix is removed) and
    extract a `RigidTransformation` from it.
    """
    # It's easier to deal with columns
    homography = homography.T
    
    # Normalize to remove any scaling factor
    homography /= norm(homography[0])
    
    trans = np.empty((4,3))
    trans[:2] = homography[:2]
    trans[2]  = np.cross(homography[0], homography[1])
    trans[3]  = homography[2]
    trans = trans.T
    
    # Constrain the first three columns to be orthogonal
    # NOTE: The decomposition used is extremely unstable. For now, just skip this step and
    #   hope for the best.
    #trans[:,0:3] = nearest_rotation_matrix(trans[:,0:3])
    
    return RigidTransformation(matrix=trans)


def homography_vector(H, i, j):
    """
    Compute a :math:`v_{ij}` vector as described by Zhang. The ordering is slightly
    different, because of the different way that `flatten_symmetric` flattens the image of
    the absolute conic.
    """
    return np.array([
        H[0][i]*H[0][j],
        H[0][i]*H[1][j] + H[1][i]*H[0][j],
        H[0][i]*H[2][j] + H[2][i]*H[0][j],
        H[1][i]*H[1][j],
        H[1][i]*H[2][j] + H[2][i]*H[1][j],
        H[2][i]*H[2][j]
    ])

def extract_camera_params(B):
    """
    Compute the camera parameters, given the *B*, the image of the absolute conic. This
    formulation assumes zero skew, and follows Sturm and Maybank (1999) rather than Zhang.
    The return value is a `CameraParameters` object.
    """
    aspect_ratio = math.sqrt(B[1,1]/B[0,0])
    center_x     = -B[2,0]/B[0,0]
    center_y     = -B[2,1]/B[1,1]
    focal_length = math.sqrt(
        (B[0,0]*B[1,1]*B[2,2] - B[1,1]*B[2,0]**2 - B[0,0]*B[2,1]**2) /
        (B[0,0]*B[1,1]**2))
    return CameraParameters(focal_length, (center_x, center_y), aspect_ratio)

def nearest_rotation_matrix(M):
    """
    Compute the orthogonal matrix which is closest to *M*. Used to constrain an estimate
    of a rotation matrix. The approach is given by Zhang in Appendix C.
    """
    # "V" is actually the (conjugate) transpose of the canonical "V" of SVD.
    U, _, V = svd(M)
    return U*V

def flatten_symmetric(M):
    r"""
    Reshape a symmetric matrix *M* into a vector which is the flattened upper triangle of
    *M*. For example, for a 3x3 matrix *M*:
    
    .. math::
       
       \begin{bmatrix}
         M_{11} & M_{21} & M_{31} \\
         M_{12} & M_{22} & M_{32} \\
         M_{13} & M_{23} & M_{33}
       \end{bmatrix}
    
    the result would be:
    
    .. math::
       
       \begin{bmatrix}
         M_{11} & M_{21} & M_{31} & M_{22} & M_{32} & M_{33}
       \end{bmatrix}^T
    
    Note that this ordering is different than, e.g., the flattening of the image of the
    absolute conic given by Zhang, but it is more natural for row-major indexing.
    """
    
    matrix_size = M.shape[0]
    vector_size = matrix_size * (1 + matrix_size) / 2
    m = np.empty((vector_size,), dtype=M.dtype)
    
    k = 0
    for i in range(matrix_size):
        for j in range(i, matrix_size):
            m[k] = M[i,j]
            k += 1
    
    return m

def unflatten_symmetric(m):
    """Perform the inverse of `flatten_symmetric`."""
    
    vector_size = m.shape[0]
    matrix_size = int((math.sqrt(1 + 8*vector_size) - 1) / 2)
    M = np.empty((matrix_size, matrix_size), dtype=m.dtype)
    
    k = 0
    for i in range(matrix_size):
        for j in range(i, matrix_size):
            M[i,j] = m[k]
            M[j,i] = m[k]
            k += 1
    
    return M
