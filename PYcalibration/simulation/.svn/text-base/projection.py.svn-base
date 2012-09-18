import numpy as np
import math
from copy import deepcopy


class EulerRotation(object):
    """
    An Euler angle parameterization of a 3D rotation using the (z,y,x) convention. This
    represents a rotation about the x axis, followed by y, followed by z. (The inverted
    ordering is confusing, but traditional. It is patterned after matrix multiplication.)
    """
    
    def __init__(self, data=None, z=None, y=None, x=None):
        """
        *data* is either a sequence (*z*, *y*, *x*) of rotations around each axis, or a
        rotation matrix from which the angles will be extracted. Alternatively, *z*, *y*,
        and *x* can be passed as keyword arguments.
        
        Examples::
        
           # As a vector of angles
           r = EulerRotation([z, y, x])
           
           # By keyword arguments
           r = EulerRotation(z=a, y=b, x=c)
           
           # As a rotation matrix
           r = EulerRotation([
             [a, b, c],
             [d, e, f],
             [g, h, i]
           ])
        """
        if data is None:
            data = [z, y, x]
        
        data = np.array(data)
        
        if data.ndim == 2:
            # Extract angles from a rotation matrix. We invert them since the angles are
            # clockwise in the calibration program.
            z = -math.atan2(-data[1,0], data[0,0])
            y = -math.asin(data[2,0])
            x = -math.atan2(-data[2,1], data[2,2])
            data = np.array([z, y, x])
        
        self.angles = data
    
    @property
    def x(self):
        return self.angles[2]
    @property
    def y(self):
        return self.angles[1]
    @property
    def z(self):
        return self.angles[0]
    @x.setter
    def x(self, value):
        self.angles[2] = value
    @y.setter
    def y(self, value):
        self.angles[1] = value
    @z.setter
    def z(self, value):
        self.angles[0] = value
    
    def __array__(self, dtype=None):
        # The angles are clockwise in the calibration program. Way to go.
        s = np.sin(-self.angles)
        c = np.cos(-self.angles)
        return np.array([
            [c[0]*c[1],
             s[0]*c[2] + c[0]*s[1]*s[2],
             s[0]*s[2] - c[0]*s[1]*c[2]],
            [-s[0]*c[1],
             c[0]*c[2] - s[0]*s[1]*s[2],
             c[0]*s[2] + s[0]*s[1]*c[2]],
            [s[1],
             -s[2]*c[1],
             c[2]*c[1]]
        ], dtype=dtype)
    
    def __copy__(self):
        return EulerRotation(self.angles)
    
    def __deepcopy__(self, memo):
        # The "shallow" copy already creates a new version of the angles array
        return self.__copy__()
    
    def __str__(self):
        return ','.join(str(n) for n in [self.x, self.y, self.z])
        

class RigidTransformation(object):
    """
    A rigid body transformation in three dimensions. Parameterized as a rotation followed
    by a translation.
    """
    
    def __init__(self, translation=None, rotation=None, matrix=None, euler=True):
        """
        *translation* and *rotation* can be given separately, or they can be extracted
        from the projective transformation *matrix*. If neither is given, they default to
        identity transformations. If *euler* is True, the rotation will be converted to an
        `EulerRotation` if it is not one already.
        """
        if translation is None:
            if matrix is None:
                translation = [0.0, 0.0, 0.0]
            else:
                translation = matrix[:3,3]
        if rotation is None:
            if matrix is None:
                rotation = np.eye(3,3)
            else:
                rotation = matrix[:3,:3]
        
        if euler and not isinstance(rotation, EulerRotation):
            rotation = EulerRotation(rotation)
        
        self.translation = translation
        self.rotation = rotation
    
    def inverse(self):
        reverse_rotation = np.asarray(self.rotation).T
        return RigidTransformation(
            np.dot(-reverse_rotation, np.asarray(self.translation)),
            reverse_rotation)
    
    def matrix(self, dtype=None):
        mat = np.eye(4, 4, dtype=dtype)
        mat[:3,:3] = np.asarray(self.rotation,    dtype=dtype)
        mat[:3,3]  = np.asarray(self.translation, dtype=dtype)
        return mat
    
    def __array__(self, dtype=None):
        return self.matrix(dtype)
    
    def __copy__(self):
        return RigidTransformation(self.translation, self.rotation)
    
    def __deepcopy__(self, memo):
        return RigidTransformation(
            deepcopy(self.translation, memo),
            deepcopy(self.rotation, memo)
        )
    
    def __str__(self):
        translation_str = ','.join(str(n) for n in np.asanyarray(self.translation).flat)
        return '%s %s' % (translation_str, self.rotation)


class CameraParameters(object):
    """
    Encapsulates the intrinsic pinhole and distortion parameters of a camera. The
    distortion uses the Brown-Conrady model with an arbitrary number of parameters.
    """
    
    def __init__(self, focal_length=1.0, center=None, aspect_ratio=1.0, skew=0.0,
                 radial=None, tangential=None):
        if center is None:
            center = [0.0, 0.0]
        if radial is None:
            radial = []
        if tangential is None:
            tangential = [0.0, 0.0]
        
        self.focal_length = focal_length
        self.center = np.asarray(center)
        self.aspect_ratio = aspect_ratio
        self.skew = skew
        self.radial_distortion = radial
        self.tangential_distortion = tangential
    
    @property
    def focal_lengths(self):
        return np.array([self.aspect_ratio * self.focal_length, self.focal_length])
    
    def __array__(self, dtype=None):
        f = self.focal_length
        return np.array([
            [f*self.aspect_ratio, self.skew, self.center[0]],
            [                0.0,         f, self.center[1]],
            [                0.0,       0.0,            1.0],
        ], dtype=dtype)
    
    def __copy__(self):
        return CameraParameters(
            self.focal_length,
            self.center,
            self.aspect_ratio,
            self.skew,
            self.radial_distortion,
            self.tangential_distortion
        )
    
    def __deepcopy__(self, memo):
        return CameraParameters(
            self.focal_length,
            deepcopy(self.center, memo),
            self.aspect_ratio,
            self.skew,
            deepcopy(self.radial_distortion, memo),
            deepcopy(self.tangential_distortion, memo)
        )
    
    def __str__(self):
        return ' '.join([
            ','.join(str(n) for n in self.focal_lengths.flat),
            ','.join(str(n) for n in np.asanyarray(self.center).flat),
            ' '.join('K' + str(coeff) for coeff in self.radial_distortion),
            ' '.join('P' + str(coeff) for coeff in self.tangential_distortion),
        ])


class Camera(object):
    """Encapsulates the pose and intrinsic parameters of a camera."""
    
    def __init__(self, pose, parameters):
        self.parameters = parameters
        self.pose = pose
        self.count = 1  # A single camera
    
    def transform(self, points):
        """
        Transform three-dimensional points in world coordinates to image points in pixels.
        *points* is a `numpy.matrix` in column vector form.
        """
        points = transform(self.pose, points)
        points = to_affine(points)
        points = distort(points,
            self.parameters.radial_distortion,
            self.parameters.tangential_distortion)
        points = to_pixels(points,
            self.parameters.center,
            self.parameters.focal_length,
            self.parameters.aspect_ratio,
            self.parameters.skew)
        return points
    
    def __copy__(self):
        return Camera(self.pose, self.parameters)
    
    def __deepcopy__(self, memo):
        return Camera(
            deepcopy(self.pose, memo),
            deepcopy(self.parameters, memo)
        )
    
    def __str__(self):
        return '%s %s' % (self.pose, self.parameters)


class StereoCamera(object):
    """
    Encapsulates a stereo rig consisting of two cameras. The rig has its own pose, which
    the individual camera poses are related to.
    """
    def __init__(self, pose, left, right):
        self.pose = pose
        self.cameras = [left, right]
        self.count = 2  # Two cameras
    
    def transform(self, points):
        # Convert points to stereo rig coordinates before passing to each camera. For
        # whatever reason, the pose is expressed in an stereo -> world transformation,
        # so we have to invert it to convert from world -> stereo.
        stereo_points = transform(self.pose.inverse(), points)
        return tuple(camera.transform(stereo_points) for camera in self.cameras)
    
    def __copy__(self):
        return StereoCamera(self.pose, *self.cameras)
    
    def __deepcopy__(self, memo):
        return StereoCamera(
            deepcopy(self.pose, memo),
            *deepcopy(self.cameras, memo)
        )


def transform(matrix, points):
    """
    Transform points using a matrix. *points* is a `numpy.ndarray` in column vector form,
    with :math:`d` rows. *matrix* is a `numpy.ndarray` with dimensions :math:`(m, n)`,
    where :math:`m` and :math:`n` are equal to :math:`p` or :math:`p + 1`. The points
    are converted to and from projective coordinates as necessary depending on the
    dimensions. The returned matrix will have the same number of rows as *points*.
    """
    
    matrix = np.asarray(matrix)
    
    vector_arity = points.shape[0]
    if vector_arity == matrix.shape[1] - 1:
        points = to_projective(points)
    
    points = np.dot(matrix, points)
    
    if vector_arity == matrix.shape[0] - 1:
        points = to_affine(points)
    return points

def to_projective(points, w=1):
    """
    Map affine coordinates to projective ones by adding a *w* coordinate to all points.
    *points* is a `numpy.matrix` of points in column vector form.
    """
    shape = (1, points.shape[1])
    return np.append(points, w * np.ones(shape, dtype=points.dtype), axis=0)

def to_affine(points):
    """
    Map projective coordinates to affine ones by dividing by (and removing) the *w*
    coordinate. *points* is a `numpy.matrix` of points in column vector form.
    """
    return points[:-1] / points[-1]

def distort(points, radial_params, tangential_params):
    """
    Add radial and tangential distortion to the given *points*, a matrix in
    column vector form. Uses the Brown-Conrady model with as many parameters as are
    given. Both *radial_params* and *tangential_params* are sequences of coefficients
    in increasing order of radial power. *tangential_params* should have at least two
    elements.
    """
    
    points = np.asarray(points)
    
    # Compute squared components and radii
    x_sq = points[0]**2
    y_sq = points[1]**2
    r_sq = x_sq + y_sq
    
    # Compute radial and tangential distortion multipliers. The tangential distortion
    # uses the third and following coefficients for the power series, while the first
    # two are used for the tangential offset.
    radial_factors = power_series(r_sq, radial_params)
    if len(tangential_params) > 2:
        tangential_factors = power_series(r_sq, tangential_params)
    else:
        tangential_factors = np.ones_like(r_sq)
    
    # Compute tangential distortion offsets
    tangential_offsets = np.zeros_like(points)
    if len(tangential_params) >= 2:
        twice_xy = 2 * points[0] * points[1]
        # The calibration program has the order of the coefficients backwards compared
        # to, well, every piece of literature out there. Surprise, surprise.
        tangential_offsets[0] = (tangential_params[1] * (r_sq + 2 * x_sq) +
                                 tangential_params[0] * twice_xy)
        tangential_offsets[1] = (tangential_params[0] * (r_sq + 2 * y_sq) +
                                 tangential_params[1] * twice_xy)
        tangential_offsets *= tangential_factors
    
    # Add distortion to the original points and convert back to a matrix
    return points * radial_factors + tangential_offsets

def to_pixels(points, center, focal_length, aspect_ratio=1.0, skew=0.0):
    r"""
    Convert image plane coordinates in world units to image coordinates in pixels.
    *points* and *center* are instances of `numpy.ndarray` in column vector form. The
    result is a matrix whose column vectors are calculated as:
    
    .. math::
    
      \mathbf{i} = \mathbf{c} + f \begin{bmatrix}
        \alpha p_x + \gamma p_y \\
        p_y
      \end{bmatrix}
    
    where :math:`\mathbf{i}` is the image point, :math:`\mathbf{p}` is the original point
    on the image plane, and :math:`f`, :math:`\alpha`, and :math:`\gamma` are the focal
    length, aspect ratio, and skew factor.
    """
    
    points[0] = points[0] * aspect_ratio + points[1] * skew
    return points * focal_length + center

def power_series(a, coefficients):
    """
    Estimate a power series on the elements of *a* up to the given *coefficients*. The
    coefficients are given in increasing order of power, and the coefficient corresponding
    to the zeroth power is assumed to be 1. Returns an `numpy.ndarray` of results for each
    element of `a`.
    """
    coefficients = list(reversed(coefficients))
    coefficients.append(1)
    return np.polyval(coefficients, a)
