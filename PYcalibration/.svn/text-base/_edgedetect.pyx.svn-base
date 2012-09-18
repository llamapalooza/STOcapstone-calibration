import numpy as np
cimport numpy as np
from cpython cimport bool
from libc.math cimport copysign

def find_zero_crossings(data, thresholds):
    """
    Find the zero crossings of a function that follow a sufficiently high (or low) value
    and precede a sufficiently low (or high) one, without any other intervening zeros or
    high/low values. This works to eliminate zero crossings caused by noise---essentially
    a roundabout high-pass filter.
    
    *data* is a one-dimensional `numpy.ndarray` with type ``'d'``. *thresholds* is a tuple
    containing the thresholds used in the filter to classify high and low values. The
    result will be a boolean array where True values are zero crossings.
    """
    # Cython doesn't support boolean NumPy arrays yet, so we have to convert it manually.
    low, high = thresholds
    return np.array(_find_zero_crossings(data, low, high), dtype=bool)

cdef inline double sign(double n):
    return copysign(1.0, n)

cdef np.ndarray[np.uint8_t] _find_zero_crossings(
        np.ndarray[np.float64_t] data, double low, double high):
    
    cdef int N
    N = len(data)
    
    cdef np.ndarray[np.uint8_t] filtered
    filtered = np.zeros((N,), dtype=np.uint8)
    
    # Enumerate the elements, looking for the sequence [high, zero, low] or
    # [low, zero, high] using a finite-state-machine approach.
    cdef double last_sign, this_sign
    cdef int last_zero
    cdef char last_thresh
    cdef int i
    last_sign = sign(data[0])
    last_zero = -1
    last_thresh = -1
    for i in range(N):
        # Zero crossings
        this_sign = sign(data[i])
        if this_sign != last_sign:
            if last_thresh > 0:
                if last_zero > 0:
                    # More than one zero in a row. Reset.
                    last_thresh = -1
                    last_zero = -1
                else:
                    # The crossing "starts" at the last value
                    last_zero = i - 1
        last_sign = this_sign
        
        # Threshold values
        if data[i] < low:
            if last_thresh == 'h' and last_zero > 0:
                filtered[last_zero] = 1
                last_zero = -1
            last_thresh = 'l'
        elif data[i] > high:
            if last_thresh == 'l' and last_zero > 0:
                filtered[last_zero] = 1
                last_zero = -1
            last_thresh = 'h'
    
    return filtered
