import numpy as np
from _edgedetect import find_zero_crossings
import sys

def moving_average(data, width=10, axis=None):
    """
    Efficiently compute a moving average with the given *width* along the specified *axis*
    of the data. Since the average cannot be computed on the ends, the returned data will
    be ``width - 1`` entries shorter than *data* along *axis*. The result will satisfy the
    invariant::
    
       moving_average(a, w)[i] == mean(a[i:i+w])
    """
    if axis is None:
        data = data.ravel()
    cum = full_cumsum(data, axis)
    s = axis_slice(axis)
    diff = cum[s[width:]] - cum[s[:-width]]
    return diff / float(width)

def threshold(data, avg_width, edge_margin, axis=None):
    """
    Compute a per-pixel edge threshold using hysteresis. The threshold is the midpoint
    between the average values in regions on either side of the center. The regions are
    *avg_width* in size and start *edge_margin* pixels from the center.
    
    For example, with an *avg_width* of 3 and an *edge_margin* of 2::
    
        o o o o o o o o o o o o o o
           |_____|   ^   |_____|
              a      c      b
    
    *a* and *b* are the two regions that are averaged when *c* is the center. The
    threshold value at *c* is the midpoint (or mean) of the means of *a* and *b*. Note
    that *c* is between two pixels. This makes the formulas more elegant, but may also
    bias the results slightly if you treat it like it is centered on a pixel. Change it if
    you feel daring.
    
    The result will be ``2 * (avg_width + edge_margin) - 1`` entries shorter than *data*
    along *axis*, and will satisfy the invariant::
    
      threshold(a, w, m)[i-m-w] == mean(mean(a[i-m-w:i-m]), mean(a[i+m:i+m+w]))
    """
    if axis is None:
        data = data.ravel()
    offset = avg_width + (2*edge_margin)
    avg = moving_average(data, avg_width, axis)
    s = axis_slice(axis)
    return (avg[s[offset:]] + avg[s[:-offset]]) / 2.0

def full_cumsum(data, axis=None, dtype=None):
    """
    A version of `numpy.cumsum` that includes the sum of the empty slice (zero). This
    makes it satisfy the invariant::
    
        cumsum(a)[i] == sum(a[:i])
    
    which is a useful property to simplify the formula for the moving average. The result
    will be one entry longer than *data* along *axis*.
    """
    
    # All we need to do is construct a result array with the appropriate type and
    # dimensions, and then feed a slice of it to cumsum, setting the rest to zero.
    
    shape = list(data.shape)
    if axis is None:
        shape[0] += 1
    else:
        shape[axis] += 1
    # Mimic cumsum's behavior with the dtype argument: use the original data type or
    # the system's native word, whichever has the greater width. (This prevents us from
    # attempting a cumulative sum using an 8-bit integer, for instance.)
    if dtype is None:
        dtype = np.promote_types(data.dtype, np.min_scalar_type(-sys.maxint))
    
    out = np.zeros(shape, dtype)
    
    s = axis_slice(axis)
    np.cumsum(data, axis, dtype, out[s[1:]])
    
    return out



def axis_slice(axis, slc=None):
    """
    Convert a simple slice into a slice tuple acting on the given axis. This is
    useful to easily promote an operation that works in one dimension into one that works
    on an arbitrary axis, acting in parallel along all the others.
    
    This function be used two ways: passing a slice object in the *slc* argument to
    construct a slice tuple immediately::
    
        subset = data[axis_slice(axis, slice(a, b))]
    
    or omitting the second argument to return a "deferred slice" object that can use
    regular slice notation to construct a slice tuple later::
    
        s = axis_slice(axis)
        subset1 = data[s[a:b]]
        subset2 = data[s[b:c]]
    """
    if slc is None:
        return AxisSlice(axis)
    if axis is None:
        return slc
    slices  = [slice(None)] * axis
    slices += [slc, Ellipsis]
    return tuple(slices)

class AxisSlice(object):
    """
    Used by `axis_slice` to defer the actual slicing operation and permit the use of the
    normal slicing syntax. See that function for details.
    """
    def __init__(self, axis):
        self.axis = axis
    def __getitem__(self, slc):
        return axis_slice(self.axis, slc)


def interpolate_zeros(data, zeros, axis):
    """
    Find the exact positions of the zero crossings of *data* along the *axis*, provided an
    array of (integer) indices which precede the actual zeros. In other words::
    
      zeros[axis] <= interpolate_zeros(data, zeros, axis)[axis] <= zeros[axis] + 1
    """
    
    adjacents = zeros.copy()
    adjacents[axis] += 1
    
    # a : (a - b) and c : 1 are corresponding sides in similar triangles.
    a = np.array(data[tuple(zeros)],     dtype=float)
    b = np.array(data[tuple(adjacents)], dtype=float)
    c = a / (a - b)
    
    exact_zeros = np.array(zeros, dtype=float)
    exact_zeros[axis] += c
    
    return exact_zeros

def detect_edges(image, axis=None, width=10, margin=5, noise_level=0.4):
    """
    Find edges along the *axis* of *image*. If *axis* is not specified, detect the edges
    along every axis and contatenate the results. *width* and *margin* are parameters for
    the `threshold` operation---see that function for details. *noise_level* controls the
    lowest signal that can be considered an edge, expressed as a multiple of the maximum
    signal. The result will be an array of multidimensional, floating-point edge positions
    with the shape (*dims*, *edges*).
    """
    
    if axis is None:
        d = image.ndim
        return np.concatenate(
            [detect_edges(image, a, width, margin, noise_level) for a in range(d)],
            axis=1)
    
    s = axis_slice(axis)
    wm = width + margin
    diff = image[s[wm:-wm+1]] - threshold(image, width, margin, axis)
    
    # Use a certain percentage of the maximum value as the noise threshold. This parameter
    # is pretty arbitrary. Figure out a way to improve on it.
    thresholds = np.array((diff.min(), diff.max()))
    thresholds *= noise_level
    
    edge_map = np.apply_along_axis(find_zero_crossings, axis, diff, thresholds)
    
    edges = np.array(edge_map.nonzero(), dtype=np.uint32)
    
    exact_edges = interpolate_zeros(diff, edges, axis)
    # Add offset due to the threshold operation
    exact_edges[axis] += wm
    return exact_edges
