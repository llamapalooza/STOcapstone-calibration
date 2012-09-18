from edgedetect import moving_average, threshold, full_cumsum
from _edgedetect import find_zero_crossings
import numpy as np
from numpy.random import rand

def test_moving_average():
    w = 3
    a = np.array(rand(50) * 50, dtype=np.int32)
    m = moving_average(a, w)
    
    assert len(m) == len(a) - w + 1
    for i in range(len(m)):
        assert m[i] == np.mean(a[i:i+w])

def test_threshold():
    aw = 3
    em = 2
    w = aw + em
    a = np.array(rand(50) * 50, dtype=np.int32)
    t = threshold(a, aw, em)
    
    assert len(t) == len(a) - 2*w + 1
    for i in range(w, len(t) + w):
        assert t[i-w] == np.mean([np.mean(a[i-w:i-em]), np.mean(a[i+em:i+w])])

def test_full_cumsum():
    a = np.array(rand(50) * 50, dtype=np.int32)
    c = full_cumsum(a)
    
    assert len(c) == len(a) + 1
    for i in range(len(c)):
        assert c[i] == sum(a[:i])

def test_full_cumsum_multidim():
    a = np.array(rand(50, 50, 50) * 50, dtype=np.int32)
    c = full_cumsum(a, axis=1)
    
    assert c.shape[0] == a.shape[0]
    assert c.shape[1] == a.shape[1] + 1
    assert c.shape[2] == a.shape[2]
    for i in range(a.shape[1]):
        assert np.all(c[:,i,:] == np.sum(a[:,:i,:], 1))

def test_find_zero_crossings():
    # Simple sign change
    a = np.array([-1, -1, 1, 1], dtype=float)
    c = find_zero_crossings(a, (0,0))
    assert(np.all(c == [False, True, False, False]))
    
    # Opposite direction
    a = np.array([1, 1, -1, -1], dtype=float)
    c = find_zero_crossings(a, (0,0))
    assert(np.all(c == [False, True, False, False]))
    
    # With threshold
    a = np.array([-1, -1, 1, 1], dtype=float)
    c = find_zero_crossings(a, (-0.5,0.5))
    assert(np.all(c == [False, True, False, False]))
    
    # Subthreshold values in between
    a = np.array([-1, -0.2, 0.2, 1], dtype=float)
    c = find_zero_crossings(a, (-0.5,0.5))
    assert(np.all(c == [False, True, False, False]))
    
    # All subthreshold
    a = np.array([-1, -1, 1, 1], dtype=float)
    c = find_zero_crossings(a, (-1.5,1.5))
    assert(np.all(c == [False, False, False, False]))
    
    # Exact zero
    a = np.array([-1, -1, 0, 1, 1], dtype=float)
    c = find_zero_crossings(a, (0,0))
    assert(np.all(c == [False, True, False, False, False]))
    