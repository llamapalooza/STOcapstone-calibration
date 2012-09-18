from simulation.projection import EulerRotation
import numpy as np

def test_euler_rotation():
    # We should be able to convert to a matrix and back and get the same numbers
    e_real = EulerRotation([0.1, 0.2, 0.3])
    e_est  = EulerRotation(np.array(e_real))
    assert np.all(e_real.angles == e_est.angles)
    
    # The matrix given should be orthogonal
    mat = np.array(e_real)
    assert np.allclose(np.dot(mat, mat.T), np.eye(3,3))