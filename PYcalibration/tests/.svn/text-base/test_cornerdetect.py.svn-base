from cornerdetect import deming, intersect_segments
import numpy as np

EPSILON = 0.1

def test_deming():
    m = 0.5
    b = 3
    
    x = np.linspace(0, 10)
    y = m*x + b
    
    x += np.random.randn(50) * 0.1
    y += np.random.randn(50) * 0.1
    
    m_e, b_e = deming(x, y)
    
    assert abs(m - m_e) < EPSILON
    assert abs(b - b_e) < EPSILON


def test_intersect_segments():
    m1 = 1
    b1 = 3
    
    m2 = 2
    b2 = -1
    
    x, y = intersect_segments((m1, m2), (b1, b2))
    
    assert x == 4
    assert y == 7