import numpy as np
import math
from numpy.dual import inv
from simulation.projection import transform
import edgedetect
from warnings import warn


def deming(x, y):
    """
    Fit a line using Deming regression (total least squares in two dimensions). Returns
    a slope-intercept tuple (*m*, *b*).
    """
    
    # S[0,0] and S[1,1] are the variances of x and y, and S[1,0] is their covariance.
    S = np.cov(x, y)
    d_var = S[1,1] - S[0,0]
    two_covar = 2 * S[1,0]
    m = (d_var + np.sqrt(d_var**2 + two_covar**2)) / two_covar
    
    mx = np.mean(x)
    my = np.mean(y)
    b = my - m*mx
    
    return m, b

def hough_transform(x, y, d_r, d_theta):
    # Calculate array of r[theta_index,point_index]
    thetas = np.arange(0, math.pi, d_theta)
    thetas = np.expand_dims(thetas, axis=1)
    rs = x*np.cos(thetas) + y*np.sin(thetas)
    
    # Round radii to nearest multiple of d_r (expressed as integer multiplier),
    # then subtract the minimum to make it zero-based
    rs = np.array(np.rint(rs / d_r), dtype=np.int32)
    r_min = rs.min()
    rs = rs - r_min
    
    # Convert to histogram of num_points[theta_index,r_index]
    num_points = np.apply_along_axis(np.bincount, 1, rs, None, rs.max() + 1)
    
    return num_points, rs, r_min
    
def hough_lines(x, y, d_r, d_theta, threshold=0.1):
    num_points, rs, r_min = hough_transform(x, y, d_r, d_theta)
    
    # Iteratively choose most significant lines and their corresponding points
    lines = []
    thetas = np.arange(num_points.shape[0])
    max = np.unravel_index(num_points.argmax(), num_points.shape)
    threshold *= num_points[max]
    while num_points[max] > threshold:
        theta, r = max
        points, = np.where(rs[theta] == r)
        lines.append((r, theta, points))

        rs[theta][points] = -1
        for point in points:
            num_points[thetas,rs[:,point]] -= 1
        num_points[max] = 0
        
        max = np.unravel_index(num_points.argmax(), num_points.shape)
    
    rs     = np.array([line[0] for line in lines], dtype=np.int32)
    thetas = np.array([line[1] for line in lines], dtype=np.int32)
    points = [line[2] for line in lines]

    rs     = (rs + r_min) * d_r
    thetas =       thetas * d_theta
    
    return rs, thetas, points

class Region:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    def contains(self, point):
        """Determine whether a point (or array of points) is inside the region."""
        # We use the bitwise & operator for compatibility with NumPy boolean arrays.
        # It works even on regular Python booleans, although the order of operations is
        # less intuitive.
        return (self.ymin <= point[0]) & (point[0] <= self.ymax) \
             & (self.xmin <= point[1]) & (point[1] <= self.xmax)
    @property
    def s(self):
        """The region represented as a slice tuple."""
        return np.s_[self.ymin:self.ymax,self.xmin:self.xmax]
    @property
    def min(self):
        """The point (ymin, xmin) represented as an array."""
        return np.array([self.ymin, self.xmin])
    @property
    def max(self):
        """The point (ymax, xmax) represented as an array."""
        return np.array([self.ymax, self.xmax])
    def __repr__(self):
        return 'Region(%r, %r, %r, %r)' % (self.xmin, self.xmax, self.ymin, self.ymax)
    
def find_target_region(corners, image_size):
    flat_corners = corners.reshape((2, -1))
    min = flat_corners.min(axis=1)
    max = flat_corners.max(axis=1)
    
    # Expand the region in order to have some data on either side of the extreme corners.
    # Make sure the regions don't go outside the image.
    # TODO: This is pretty arbitrary. Base it off a rough estimate of horizontal and
    #   vertical spans of a square on the checkerboard?
    margin = 0.3 * (max - min)
    margin = margin.astype(np.int)
    xmin, ymin = np.maximum(min - margin, [0, 0])
    xmax, ymax = np.minimum(max + margin, [image_size[1], image_size[0]])
    
    return Region(xmin, xmax, ymin, ymax)

def assign_segments(points, homography):
    # Use the homography to project image points back onto their board locations. If we
    # assume that the corners are on lattice points, then assigning the points to line
    # segments becomes as simple as rounding their coordinates to the nearest integer.
    # The choice of vertical or horizontal segment depends on the distance to the lattice
    # point on each axis
    
    points = back_project(homography, points)
    
    nearest_corners = np.rint(points)
    distances = np.abs(points - nearest_corners)
    directions = np.argmin(distances, axis=0)
    
    return nearest_corners, directions
    
def fit_segments(points, nearest_corners, directions, corner_shape):
    shape = (2,) + corner_shape
    ms = np.empty(shape)
    bs = np.empty(shape)
    
    for row, col in np.ndindex(corner_shape):
        on_corner = np.all(nearest_corners == [[row], [col]], axis=0)
        for direction in range(2):
            in_segment = on_corner & (directions == direction)
            xs, ys = points[:,in_segment]
            if len(xs) > 0:
                m, b = deming(xs, ys)
            else:
                m, b = None, None
            ms[direction,row,col] = m
            bs[direction,row,col] = b
    
    return ms, bs

def intersect_segments(ms, bs):
    x = (bs[0] - bs[1]) / (ms[1] - ms[0])
    y = ms[0]*x + bs[0]
    return x, y

def refine_corners(image, rough_corners, homography):
    # Find edges within the target region and convert them back to full image coordinates
    region = find_target_region(rough_corners, image.shape)
    y, x = edgedetect.detect_edges(image[region.s])
    y += region.min[0]
    x += region.min[1]
    edges = np.array([x, y])
    
    new_corners = find_corners(edges, rough_corners.shape[1:], homography)
    
    # Find any corners that could not be determined and replace them with their rough
    # estimates
    nans = np.isnan(new_corners).any(axis=0)
    if nans.any():
        count = len(nans.nonzero()[0])
        warn(
          "The positions of %d corners could not be determined, probably because no "
          "edges were found in the area. Try tuning the edge noise threshold and "
          "checking that the corner estimates are sane." % count)
        
        new_corners[:,nans] = rough_corners[:,nans]
    
    return new_corners
        

def find_corners(edges, corner_shape, homography):
    nearest_corners, directions = assign_segments(edges, homography)
    ms, bs = fit_segments(edges, nearest_corners, directions, corner_shape)
    xs, ys = intersect_segments(ms, bs)
    return np.array([xs, ys])

def back_project(homography, points):
    return transform(inv(homography), points)

def reproject(homography, points):
    return transform(homography, back_project(homography, points))