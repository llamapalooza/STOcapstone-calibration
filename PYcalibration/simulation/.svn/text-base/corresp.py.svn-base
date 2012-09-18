from projection import transform, StereoCamera
from StringIO import StringIO
from copy import deepcopy
from collections import defaultdict
import numpy as np


class Correspondences(object):
    """Holds camera parameter, pose, and point correspondence data for a set of images."""
    def __init__(self, cameras, poses, measurements, target_trans, target_scale=1.0):
        #: A pair of cameras in a stereo rig, instances of `Camera`. Their poses are
        #: relative to the stereo rig coordinates.
        self.cameras = cameras
        #: A dictionary mapping the names of image pairs to the pose of the stereo rig.
        self.poses = poses
        #: Measured image points indexed by
        #: ``measurements[target_num][target_point][image_name]``. The target points are
        #: represented as tuples to make them hashable.
        self.measurements = measurements
        #: A matrix-like object describing the coordinates of the first target relative
        #: to the second.
        self.target_transformation = target_trans
        #: The scale of the first target relative to the second.
        self.target_scale = target_scale
    
    @property
    def images(self):
        """The set of images whose correspondences are measured."""
        return self.poses.keys()
    @property
    def points(self):
        """A pair containing interest point matrices for each target."""
        return tuple(
            np.array(target_points.keys()).T for target_points in self.measurements)
    
    def project(self):
        """
        Project 3D points based on camera parameters. These theoretical projections can
        be compared to the measured correspondences to yield the reprojection error.
        Returns a pair of dictionaries with the same structure as :attr:`measurements`.
        """
        projections = (defaultdict(dict), defaultdict(dict))
        for target_num, target_points in enumerate(self.points):
            # The transformation applies to the first target, and only the first target.
            # The second target's points are in world coordinates.
            if target_num == 0:
                world_points = transform(self.target_transformation, target_points)
                world_points *= self.target_scale
            else:
                world_points = target_points
            
            for image_pair_name, stereo_pose in self.poses.items():
                stereo_rig = StereoCamera(stereo_pose, *self.cameras)
                # This is actually a *pair* of point matrices for each camera
                camera_points = stereo_rig.transform(world_points)
                
                image_names = [image_pair_name + 'L', image_pair_name + 'R']
                
                # Add each corresponding (target point, camera point) pair to the
                # projection matrix
                for point_num in range(target_points.shape[1]):
                    target_point = tuple(target_points[:,point_num].flat)
                    for camera_num, image_name in enumerate(image_names):
                        camera_point = camera_points[camera_num][:,point_num]
                        projections[target_num][target_point][image_name] = camera_point
        
        return projections
    
    def dump(self, output=None):
        """
        Dump correspondence data in the "corresp" format. If *output* is an open stream,
        the data will be written to it. Otherwise, the data is returned as a string.
        This method assumes that all transformations are instances of
        `RigidTransformation` with rotations represented by `EulerRotation`.
        """
        return_data = (output is None)
        if return_data:
            output = StringIO()
        
        total_points = len(self.points[0].T) + len(self.points[1].T)
        total_images = len(self.images) * 2
        output.write('CORRESPONDENCES %d %d\n' % (total_images, total_points))
        output.write('connections %f %s\n' %
            (self.target_scale, self.target_transformation))
        output.write('basePairL %s\n' % self.cameras[0])
        output.write('basePairR %s\n' % self.cameras[1])
        
        for image_name, rig_pose in sorted(self.poses.items()):
            output.write('%s\t%s\n' % (image_name, rig_pose))
        
        for target_corresps in self.measurements:
            for target_point, image_points in sorted(target_corresps.items()):
                output.write(_comma_sep(target_point) + '\t')
                for image_name, image_point in sorted(image_points.items()):
                    output.write(' %s %s' % (image_name, _comma_sep(image_point)))
                output.write('\n')
            output.write('\n')
        
        if return_data:
            return str(output)
    
    def __copy__(self):
        return Correspondences(
            self.cameras,
            self.poses,
            self.measurements,
            self.target_transformation,
            self.target_scale
        )
    
    def __deepcopy__(self, memo):
        return Correspondences(
            deepcopy(self.cameras, memo),
            deepcopy(self.poses, memo),
            deepcopy(self.measurements, memo),
            deepcopy(self.target_transformation, memo),
            self.target_scale
        )


def _comma_sep(vector):
    return ",".join(str(n) for n in np.asanyarray(vector).flat)
