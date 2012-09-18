#!/usr/bin/env python

import sys
from os.path import basename, join
from glob import glob
import logging

import numpy as np
import cv2

from calibration import estimate_camera, factor_target_pose, factor_camera_pose
import cornerdetect
from simulation.projection import Camera, RigidTransformation
from simulation.corresp import Correspondences

logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s')
if hasattr(logging, 'captureWarnings'):
    logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(logging.INFO)

CORNERS_FILE    = '%s%s.%s.npy'
HOMOGRAPHY_FILE = '%s%s.%shom.txt'
IMAGE_FILE      = '%s%s.ppm'

EDGE_NOISE_LEVEL = 0.2

CAMERAS = ['L', 'R']
TARGETS = ['R', 'B']

def main():
    # This layout follows the one in /home/palantir/calib. Adjust as necessary, or add
    # command-line options.
    folder = sys.argv[1]
    corners_folder = join(folder, 'corners')
    images_folder  = join(folder, 'ppm')
    
    base_names = [basename(fn)[:-7] for fn in glob(join(corners_folder, '*L.B.npy'))]
    #  image_names[camera_num][base_name]
    #      corners[camera_num][target_num][base_name]
    # homographies[camera_num][target_num][base_name]
    image_names  = find_images(join(images_folder, IMAGE_FILE), base_names)
    corners      = load_target_data(join(corners_folder, CORNERS_FILE),    base_names)
    homographies = load_target_data(join(corners_folder, HOMOGRAPHY_FILE), base_names, np.loadtxt)
    
    # The first target's homography is just an extra rigid transformation applied
    # before the second's. We will estimate it using the homographies from one image.
    first_target_pose = factor_target_pose(
        homographies[0][1][base_names[0]],
        homographies[0][0][base_names[0]])
    
    params, poses = zip(*[estimate_camera(camera_hs[0].values()) for camera_hs in homographies])
    
    # The first camera's pose is the stereo rig's pose. The second's pose is an extra
    # rigid transformation applied after the first's. Again, we estimate this using the
    # transformations from one image only.
    stereo_poses = dict(zip(base_names, poses[0]))
    second_camera_pose = factor_camera_pose(poses[0][0], poses[1][0])
    cameras = [
        Camera(RigidTransformation(), params[0]),
        Camera(second_camera_pose,    params[1])
    ]
    
    # Refine the corner measurements
    N = 2 * len(base_names)
    log.info("Finding corners in %d files:", N)
    i = 0
    for base_name in base_names:
        for camera, camera_name in enumerate(CAMERAS):
            pct = float(i) / N * 100
            log.info("  %s%s (%d of %d, %.1f%%)", base_name, camera_name, i+1, N, pct)
            i += 1
            
            r, g, b = np.rollaxis(cv2.imread(image_names[camera][base_name]), 2)
            

            for target in range(len(TARGETS)):
                homography = homographies[camera][target][base_name]
                rough_corners = corners[camera][target][base_name]
            
                # Change the shape of corners from corners[column,row,axis] to
                # corners[axis,row,column], where axis is 0 for the x axis or 1 for y
                rough_corners = rough_corners.swapaxes(0, 2)
                
                corners[camera][target][base_name] = [
                    cornerdetect.refine_corners(channel, rough_corners, homography)
                    for channel in [r, g, b]]
    log.info("Done finding corners.")
    
    # Reshuffle the whole mess into the format wanted by Correspondences.measurements
    measurements = []
    for target in range(len(TARGETS)):
        target_shape = corners[0][target][base_names[0]][0].shape[1:]
        
        # NOTE: Adding per-channel locations is currently just a hack where we pretend
        #   there are six targets instead of two. Adding better support in the
        #   Correspondences class wouldn't be a bad idea.
        for channel in range(3):
            target_measurements = {}
            for row, col in np.ndindex(target_shape):
                corner_measurements = {}
                for base_name in base_names:
                    for camera, camera_name in enumerate(CAMERAS):
                        corner_measurements[base_name + camera_name] = \
                            corners[camera][target][base_name][channel][:,row,col]
                target_measurements[(col, row, 0)] = corner_measurements
            measurements.append(target_measurements)

    # Barf
    corresp = Correspondences(cameras, stereo_poses, measurements, first_target_pose)
    corresp.dump(sys.stdout)

def load_target_data(pattern, base_names, load=np.load):
    return [[dict((base_name, load(pattern % (base_name, camera_name, target_name)))
                  for base_name in base_names)
                  for target_name in TARGETS]
                  for camera_name in CAMERAS]

def find_images(pattern, base_names):
    return [dict((base_name, pattern % (base_name, camera_name))
                 for base_name in base_names)
                 for camera_name in CAMERAS]

if __name__ == '__main__':
    main()
