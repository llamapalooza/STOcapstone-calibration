from calibration import *
from simulation.projection import EulerRotation, CameraParameters, RigidTransformation
from simulation.simulation import random_angle
import numpy as np

def test_nearest_rotation_matrix():
    # If we use an actual rotation matrix, we should get it back
    r_real = np.array(EulerRotation([0.1, 0.2, 0.3]))
    r_est = nearest_rotation_matrix(r_real)
    assert np.allclose(r_real, r_est)

def test_flatten_symmetric():
    # Flatten and unflatten should be inverse operations
    M = np.empty((4, 4))
    for i in range(4):
        for j in range(i, 4):
            M[i,j] = M[j,i] = np.random.random()
    M2 = unflatten_symmetric(flatten_symmetric(M))
    assert np.all(M == M2)

def test_estimate_camera():
    # Test the accuracy under ideal conditions (perfect model, no noise)
    
    params = np.array(CameraParameters(1500, [10,15], 0.998))
    poses = [
        RigidTransformation(
            np.random.multivariate_normal([-0.2, 0.0, 0.0], 0.01 * np.eye(3, 3)),
            EulerRotation(
                z = random_angle(stdev=0.003),
                y = random_angle(stdev=0.01),
                x = random_angle(stdev=0.003)
            )
        )
        for _ in range(10)
    ]
     
    homographies = [np.dot(params, np.array(pose)[:3]) for pose in poses]
    homographies = [np.concatenate((H[:,:2], H[:,3:4]), axis=1) for H in homographies]
    
    params_est, poses_est = estimate_camera(homographies)
    
    assert np.allclose(params, params_est)
    assert all(np.allclose(pose, pose_est) for pose, pose_est in zip(poses, poses_est))