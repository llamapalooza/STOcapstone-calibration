#!/usr/bin/env python2.7

from projection import (to_projective, Camera, StereoCamera, CameraParameters,
    RigidTransformation, EulerRotation)
from corresp import Correspondences
import numpy as np
import cv2
from numpy import random
from math import fmod, pi
from copy import deepcopy


IMAGE_SIZE = np.array([3000, 2000])
ASPECT_RATIO = 1.0
SKEW = 0.0
TARGET_SCALE = 1.0
NUM_RADIAL = 3


def main():
    true_corresp = generate_calibration_data()
    with open('true.corresp', 'w+') as file:
        true_corresp.dump(file)
    
    noisy_corresp = deepcopy(true_corresp)
    add_calibration_noise(noisy_corresp)
    with open('noisy.corresp', 'w+') as file:
        noisy_corresp.dump(file)
    
    write_images(true_corresp)


def generate_calibration_data(num_image_pairs = 6):
    """
    Simulate calibration data by generating a random set of camera parameters, poses, and
    target positions and projecting point correspondences.
    """
    target_points = [
        to_projective(gen_point_mesh().reshape((-1, 2)), w=0.0) for _ in range(2)]
    target_scale = TARGET_SCALE
    target_transform = RigidTransformation(
        np.array(random.multivariate_normal([-1.0, 0.0, -1.0], 0.05 * np.eye(3, 3))),
        EulerRotation(
            z = random_angle(stdev=0.01),
            y = random_angle(mean=-1.0),
            x = random_angle()
        )
    )
    
    camera_params = [gen_camera_params(), gen_camera_params()]
    camera_poses = [
        # The first camera stays at the origin of the stereo rig coordinate system
        RigidTransformation(),
        RigidTransformation(
            random.multivariate_normal([-0.2, 0.0, 0.0], 0.01 * np.eye(3, 3)),
            EulerRotation(
                z = random_angle(stdev=0.003),
                y = random_angle(stdev=0.01),
                x = random_angle(stdev=0.003)
            )
        )
    ]
    cameras = [Camera(camera_poses[i], camera_params[i]) for i in range(2)]
    # The pose is ignored in the correspondence data, so we don't have to pass it here
    stereo_rig = StereoCamera(None, *cameras)
    
    poses = {}
    for image_pair_num in range(num_image_pairs):
        image_pair_name = "IMG_%03d" % image_pair_num
        poses[image_pair_name] = RigidTransformation(
            random.multivariate_normal([0.0, 0.0, -5.0], 0.1 * np.eye(3, 3)),
            EulerRotation(
                z = random_angle(stdev=0.01),
                y = random_angle(stdev=0.1),
                x = random_angle(stdev=0.1)
            )
        )
    
    # We won't pass in any measurements, and instead let it calculate the true values for
    # us. We still have to specify the world points to project. Note that we have to
    # convert to row vector form to iterate over points in a straightforward way, and we
    # have to convert the points to tuples to make them hashable.
    points = [dict((tuple(point.flat), {}) for point in points.T) for points in target_points]
    corresp = Correspondences(stereo_rig, poses, points, target_transform,
        target_scale)
    corresp.measurements = corresp.project()
    return corresp

def random_angle(mean=0.0, stdev=0.04):
    r"""
    Generate a random angle clamped to a range slightly narrower than :math:`\left(
    -\frac{\pi}{2}, \frac{\pi}{2} \right)` to avoid singularities in an Euler angle
    representation. Within that range, the angles will follow the normal distribution
    with the given mean and standard deviation.
    """
    return fmod(random.normal(mean, stdev), 0.45 * pi)

def gen_camera_params():
    """Generate a random set of camera parameters."""
    return CameraParameters(
        focal_length = random.normal(IMAGE_SIZE[0], IMAGE_SIZE[0] / 10.0),
        center       = random.normal(1.0, 0.05, (2,)) * IMAGE_SIZE / 2.0,
        aspect_ratio = ASPECT_RATIO,
        skew         = SKEW,
        radial       = [random.normal(0.0, 10**-(n+1)) for n in range(NUM_RADIAL)],
        tangential   = random.normal(0.01, 0.01, (2,))
    )

def gen_point_mesh(width = 8, height = 8):
    """Generate a (*width* x *height*) mesh of points equally spaced in two dimensions."""
    return np.array(np.meshgrid(
        np.linspace(0.0, 1.0, width),
        np.linspace(0.0, 1.0, height)
    )).swapaxes(0, 2)


def add_calibration_noise(corresp):
    """
    Add noise to the camera parameters, poses, and projected image points in a set of
    calibration data, modifying the original object.
    """
    
    add_transformation_noise(corresp.target_transformation)
    corresp.target_scale += random.normal(0.0, 0.1)
    
    for camera_num, camera in enumerate(corresp.camera.cameras):
        # Let the first camera stay at the origin of the stereo rig
        if camera_num == 1:
            add_transformation_noise(camera.pose)
        add_parameter_noise(camera.parameters)
    
    for stereo_rig_pose in corresp.poses.values():
        add_transformation_noise(stereo_rig_pose)
    
    for target_points in corresp.measurements:
        for image_points in target_points.values():
            for image_point in image_points.values():
                add_point_noise(image_point)
    
def add_transformation_noise(transformation):
    """
    Add random noise to a rigid body transformation with an Euler angle rotation,
    modifying the original object.
    """
    transformation.translation += random.normal(0.0, 0.1, (3,))
    transformation.rotation.x  += random.normal(0.0, 0.01)
    transformation.rotation.y  += random.normal(0.0, 0.01)
    transformation.rotation.z  += random.normal(0.0, 0.01)

def add_parameter_noise(params):
    """Add noise to a set of camera parameters, modifying the original object."""
    params.focal_length += random.normal(0.0, 0.1 * IMAGE_SIZE[0])
    params.center += random.normal(0.0, 0.1 * IMAGE_SIZE[0])
    params.aspect_ratio += random.normal(0.0, 0.1)
    # Skew isn't taken into account by the calibration program we're using
    #params.skew += random.normal(0.0, 0.1)
    params.radial_distortion += [
        random.normal(0.0, 0.3 * 10**-(n+1)) for n in range(NUM_RADIAL)]
    params.tangential_distortion += random.normal(0.0, 0.003, (2,))

def add_point_noise(point):
    """Add noise to an image point vector, modifying the original."""
    point += random.normal(0.0, 0.3, (2,))


def write_images(corresp):
    """
    Create reference image files based on projected point data in an set of calibration
    data.
    """
    for image_pair_name in corresp.images:
        image_names = [image_pair_name + 'L', image_pair_name + 'R']
        for image_name in image_names:
            image = np.empty((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.ubyte)
            image.fill(240)  # Slightly off-white to make the image boundary clear
            
            for target_measurements in corresp.measurements:
                points = np.array(np.hstack(
                    points[image_name] for points in target_measurements.values()))
                points = points.astype(np.int32)
                
                zero = np.zeros((2,))
                size = IMAGE_SIZE
                for point in points.T:
                    if np.all(point < zero) or np.all(point >= size):
                        continue
                    cv2.circle(image, tuple(point), 3, 0, -1) 
            
            cv2.imwrite(image_name + '.pgm', image)


# Run main function if invoked as script
if __name__ == '__main__':
    main()
