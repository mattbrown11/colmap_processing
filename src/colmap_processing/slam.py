#!/usr/bin/env python
"""
ckwg +31
Copyright 2020 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Library handling projection operations of a standard camera model.

Note: the image coordiante system has its origin at the center of the top left
pixel.

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import cv2
import copy
import os
import time
from random import shuffle
import gtsam
from gtsam import (Cal3_S2, DoglegOptimizer,
                   GenericProjectionFactorCal3_S2, Marginals,
                   NonlinearFactorGraph, PinholeCameraCal3_S2, Point3,
                   Pose3, PriorFactorPoint3, PriorFactorPose3, Rot3, Values)
from gtsam import symbol_shorthand
L = symbol_shorthand.L
X = symbol_shorthand.X

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Repository imports.
from colmap_processing.platform_pose import PlatformPoseInterp
from colmap_processing.calibration import horn
from colmap_processing.rotations import quaternion_from_matrix, \
    quaternion_matrix
from colmap_processing.colmap_interface import read_images_binary, \
    read_points3D_binary, read_cameras_binary, standard_cameras_from_colmap
from colmap_processing.colmap_interface import Image as ColmapImage


def fit_pinhole_camera(cm):
    """Solve for a best-matching pinhole (distortion-free) camera model.

    For SLAM problems where we already have an accurate model for the camera's
    intrinsic parameters, we only want to optimize camera pose and world
    geometry at runtime. Therefore, it is convenient to warp image coordinates
    within the original image to appear as if they came from a pinhole
    (distrortion-free) camera. Since we might do reprojection error
    calculations, we want to maintain the size of a 1x1 pixel projected into
    the new pinhole camera to be as closer to 1x1.

    Parameters
    ----------
    cm : camera_models.StandardCamera
        Camera model with distortion.

    Returns
    -------
    cm_pinhole : camera_models.StandardCamera
        Camera model without distortion.
    """
    cm_pinhole = copy.deepcopy(cm)
    cm_pinhole.dist = 0
    im_pts = cm.points_along_image_border(1000)

    ray_pos, ray_dir = cm.unproject(im_pts, 0)
    im_pts2 = cm_pinhole.project(ray_pos + ray_dir, 0)

    #plt.plot(im_pts[0],im_pts[1]); plt.plot(im_pts2[0],im_pts2[1])

    dx1 = - min([min(im_pts2[0]), 0])
    dy1 = - min([min(im_pts2[1]), 0])
    dx2 =  max([max(im_pts2[0]) - cm.width, 0])
    dy2 = max([max(im_pts2[1]) - cm.height, 0])
    cm_pinhole.cx = cm_pinhole.cx + dx1
    cm_pinhole.cy = cm_pinhole.cy + dy1
    cm_pinhole.width = cm_pinhole.width + int(np.ceil(dx1 + dx2))
    cm_pinhole.height = cm_pinhole.height + int(np.ceil(dy1 + dy2))

    return cm_pinhole


def draw_keypoints(image, pts, radius=2, color=(255, 0, 0),
                   thickness=1, copy=False):
    """
    :param image: Image/
    :type image: Numpy image

    :param pts: Keypoints to be drawn in the image.
    :type pts: Numpy array num_pts x 2
    """
    pts = np.round(pts).astype(int)

    if len(color) == 3 and image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif copy:
        image = image.copy()

    for pt in pts:
        cv2.circle(image, (pt[0], pt[1]), radius, color, thickness)

    return image


def map_to_pinhole_problem(cm, im_pts_at_time):
    cm_pinhole = fit_pinhole_camera(cm)

    for t in im_pts_at_time:
        im_pts, point3D_ind = im_pts_at_time[t]
        ray_pos, ray_dir = cm.unproject(im_pts, t)
        im_pts2 = cm_pinhole.project(ray_pos + ray_dir*100, t)
        im_pts_at_time[t] = im_pts2, point3D_ind

    return cm_pinhole, im_pts_at_time


def reprojection_error(cm, im_pts_at_time, wrld_pts, plot_results=False):
    """Calculate mean reprojection error in pixels.
    """
    err = []
    image_times = sorted(list(im_pts_at_time.keys()))
    for t in image_times:
        im_pts, point3D_ind = im_pts_at_time[t]
        err.append(np.mean(np.sum((im_pts - cm.project(wrld_pts[:, point3D_ind], t))**2, axis=0)))

    if plot_results:
        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        plt.plot(image_times, err)
        plt.xlabel('Time (s)', fontsize=40)
        plt.ylabel('Image Mean Error (pixels)', fontsize=40)
        print('Time with max error', image_times[np.argmax(err)])

    return np.mean(err)


def rescale_sfm_to_ins(cm, ins, image_times, wrld_pts):
    # Rescale using the ins.
    pts1 = []
    pts2 = []
    for i, t  in enumerate(image_times):
        P = cm.get_camera_pose(t)
        R = P[:, :3]
        pts1.append(np.dot(-R.T, P[:, 3]))
        pts2.append(ins.pose(t)[0])

    pts1 = np.array(pts1).T
    pts2 = np.array(pts2).T
    s, R, trans = horn(pts1, pts2, fit_scale=True, fit_translation=True)

    wrld_pts2 = np.dot(R, s*wrld_pts) + np.atleast_2d(trans).T

    cm2 = copy.deepcopy(cm)
    ppp = cm.platform_pose_provider
    ppp2 = PlatformPoseInterp(ppp.lat0, ppp.lon0, ppp.h0)
    for i in range(len(image_times)):
        t = image_times[i]
        pos, quat = ppp.pose(t)
        pos = np.dot(R, s*pos) + trans
        R0 = quaternion_matrix(quat)[:3, :3]
        R1 = np.dot(R, R0)
        quat = quaternion_from_matrix(R1)
        ppp2.add_to_pose_time_series(t, pos, quat)

    cm2.platform_pose_provider = ppp2

    return cm2, wrld_pts2


def read_colmap_results(recon_dir, use_camera_id=1, max_images=None,
                        max_image_pts=None, min_track_len=3):
    """Read colmap bin data and return camera and image with 3d point pairs.

    Parameters
    ----------
    recon_dir : str
        Direction in which to find colmap 'images.bin', 'cameras.bin', and
        'points3D.bin' files.

    use_camera_id : int | None
        If Colmap didn't use a single camera for all images, we can force to
        use one camera model for all images. This sets the index of the desired
        camera to use.

    Returns
    -------
    sfm_cm : camera_models.StandardCamera
        Camera model that accepts

    im_pts_at_time : dict
        Dictionary taking image time and returning the tuple
        (im_pts, point3D_ind), where 'im_pts' is a 2 x N array of image
        coordinates and that are associated with wrld_pts[:, point3D_ind].

    wrld_pts : 3 x N array
        3-D coordinates of world points that are correlated with image points.

    image_name : dict
        Dictionary taking image time (s) and returning image name.


    """
    # ------------------- Read Existing Colmap Reconstruction ----------------
    # Read in the Colmap details of all images.
    images_bin_fname = '%s/images.bin' % recon_dir
    colmap_images = read_images_binary(images_bin_fname)
    camera_bin_fname = '%s/cameras.bin' % recon_dir
    colmap_cameras = read_cameras_binary(camera_bin_fname)
    points_bin_fname = '%s/points3D.bin' % recon_dir
    points3d = read_points3D_binary(points_bin_fname)

    if max_images is not None:
        colmap_images0 = colmap_images
        colmap_images = {}
        for t in sorted(list(colmap_images0.keys()))[:max_images]:
            colmap_images[t] = colmap_images0[t]

    use_camera_id = 1
    image_times = {}
    for ind in colmap_images:
        colmap_images[ind] = ColmapImage(colmap_images[ind].id,
                                         colmap_images[ind].qvec,
                                         colmap_images[ind].tvec,
                                         use_camera_id,
                                         colmap_images[ind].name,
                                         colmap_images[ind].xys,
                                         colmap_images[ind].point3D_ids)
        img_fname = os.path.split(colmap_images[ind].name)[1]
        image_times[ind] = float(os.path.splitext(img_fname)[0])/1000000

    sfm_cm = standard_cameras_from_colmap(colmap_cameras, colmap_images,
                                          image_times)[use_camera_id]

    wrld_pts = [points3d[i].xyz if i in points3d else None
                for i in range(max(points3d.keys()) + 1)]

    used_3d = np.zeros(len(wrld_pts), dtype=int)

    im_pts_at_time = {}
    image_name = {}
    for image_num in colmap_images:
        image = colmap_images[image_num]
        xys = image.xys
        point3D_ind = image.point3D_ids
        ind = point3D_ind != -1
        point3D_ind = point3D_ind[ind]
        xys = xys[ind]

        if max_image_pts is not None and len(xys) > max_image_pts:
            ind = list(range(len(xys)))
            shuffle(ind)
            ind = sorted(ind[:max_image_pts])
            xys = xys[ind]
            point3D_ind = point3D_ind[ind]

        used_3d[point3D_ind] += 1

        img_fname = os.path.split(image.name)[1]
        t = float(os.path.splitext(img_fname)[0])/1000000
        im_pts_at_time[t] = (xys.T, point3D_ind)
        image_name[t] = image.name

    # Remove ununsed 3d points.
    ind = np.nonzero(used_3d >= min_track_len)[0]
    orig_map = np.full(len(wrld_pts), -1, dtype=int)
    orig_map[ind] = range(len(ind))
    wrld_pts = np.array([wrld_pts[i] for i in ind]).T

    for t in im_pts_at_time:
        xys, point3D_ind = im_pts_at_time[t]
        point3D_ind = orig_map[point3D_ind]
        ind = point3D_ind >= 0
        xys = xys[:, ind]
        point3D_ind = point3D_ind[ind]
        im_pts_at_time[t] = xys, point3D_ind

    return sfm_cm, im_pts_at_time, wrld_pts, image_name


class OfflineSLAM(object):
    def __init__(self, cm, min_gps_std=None, min_orientation_std=None,
                 pixel_sigma=10, robust_pixels_k=4, robust_ins_k=4):
        """
        :param cm: Camera model.
        :type cm: camera_models.StandardCamera
        """
        # Define the camera calibration parameters
        self.gtsam_camera = Cal3_S2(cm.fx, cm.fy, 0.0, cm.cx, cm.cy)

        self.cm = cm
        self.min_gps_std = min_gps_std
        self.min_orientation_std = min_orientation_std
        self.pixel_sigma = pixel_sigma
        self.robust_ins_k = robust_ins_k
        self.robust_pixels_k = robust_pixels_k

        # Rotation matrix that moves vectors from ins coordinate system into camera
        # coordinate system.
        Rcam = cm.get_camera_pose(0)[:, :3]
        Rins = quaternion_matrix(cm.platform_pose_provider.pose(0)[1])[:3, :3].T
        self.Rins_to_cam = np.dot(Rcam, Rins.T)

        self.reset_graph()

    def __str__(self):
        string = ['OfflineSLAM\n']
        string.append('min_gps_std: %s\n' % str(self.min_gps_std))
        string.append('min_orientation_std: %s\n' % str(self.min_orientation_std))
        string.append('pixel_sigma: %s\n' % str(self.pixel_sigma))
        string.append('robust_ins_k: %s\n' % str(self.robust_ins_k))
        string.append('robust_pixels_k: %s\n' % str(self.robust_pixels_k))
        return ''.join(string)

    def __repr__(self):
        return self.__str__()

    def reset_graph(self):
        self.graph = NonlinearFactorGraph()
        self.num_landmarks = None

    def define_problem(self, im_pts_at_time, wrld_pts, cm_sfm=None):
        print('Calculating initial poses and adding pose prior estimates')
        tic = time.time()

        image_times = sorted(list(im_pts_at_time.keys()))
        poses = []

        # Create the set of ground-truth landmarks
        points3d = wrld_pts.T
        wrld_pts_used = np.zeros(len(points3d), dtype=int)

        min_gps_std = self.min_gps_std
        if not hasattr(min_gps_std, "__len__"):
            min_gps_std = [min_gps_std, min_gps_std, min_gps_std]

        min_orientation_std = self.min_orientation_std
        if not hasattr(min_orientation_std, "__len__"):
            min_orientation_std = [min_orientation_std, min_orientation_std,
                                   min_orientation_std]


        for i, t in enumerate(image_times):
            # Loop over all images

            # This image has 'N' points.
            im_pts, point3D_ind = im_pts_at_time[t]
            N = im_pts.shape[1]

            # ------------------- Add pose prior from INS  -------------------
            if cm_sfm is not None:
                P = cm_sfm.get_camera_pose(t)
            else:
                P = self.cm.get_camera_pose(t)

            Rcam = P[:, :3]
            pos = np.dot(-Rcam.T, P[:, 3])
            poses.append(Pose3(Rot3(Rcam.T), pos))

            # Provide a prior derived from the INS.
            pos_ins, quat_ins, std = self.cm.platform_pose_provider.pose(t, return_std=True)
            std_e, std_n, std_u, std_y, std_p, std_r = std

            if self.min_gps_std is not None:
                std_e = max([std_e, min_gps_std[0]])
                std_n = max([std_n, min_gps_std[1]])
                std_u = max([std_u, min_gps_std[2]])

            if self.min_orientation_std is not None:
                std_y = max([std_y, min_orientation_std[0]])
                std_p = max([std_p, min_orientation_std[1]])
                std_r = max([std_r, min_orientation_std[2]])

            Rins = quaternion_matrix(quat_ins)[:3, :3].T
            Rcam_from_ins = np.dot(self.Rins_to_cam, Rins)

            # Vector defined within the INS coordinate system representing the
            # uncertainty of its orientation
            rot_ind_std = np.array([std_r, std_p, std_y])

            # Rotate into camera coordinate system
            std_r, std_p, std_y = np.abs(np.dot(self.Rins_to_cam, rot_ind_std))

            #std_r = std_p = std_y = 1

            # Pose standard deviation defined in roll (rad), pitch (rad), yaw (rad), x
            # (m), y (m), z (m).
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([std_r, std_p,
                                                                    std_y, std_e,
                                                                    std_n, std_u]))

            if self.robust_ins_k is not None:
                huber = gtsam.noiseModel.mEstimator.Huber.Create(self.robust_ins_k)
                pose_noise = gtsam.noiseModel.Robust.Create(huber, pose_noise)

            for _ in range(N):
                # We re-add the pose one for each image measurement so that
                # they have equal influence on solution.
                pose = Pose3(Rot3(Rcam_from_ins.T), pos_ins)
                factor = PriorFactorPose3(X(i), pose, pose_noise)
                self.graph.push_back(factor)


            # -------------- Add image measurements for this image -----------

            measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.pixel_sigma)  # one pixel in u and v



            # The idea here is that we want to balance the contribution to the overall
            # likelihood between an INS factor and the factors from image measurements
            # on one image. For delta^2 loss, where delta is a normalized version of
            # some paramenter normalized by its standard deviation, when delta is 1
            # (i.e., 1-sigma) dLoss/ddelta is 2. If we similarly express the pixel
            # error in units of sigma, we derivative of the sum of the loss over all
            # pixel measurements in an image to equal 2. So, we can use Huber loss to
            # effect this by chosing Huber k such that when all pixels errors are at
            # 1-sigma, the sum of the derivatives equals 2.
            # huber loss = (delta)^2/2 up to k with loss gradient x. So at k, slope is
            # k forever. So, k = 2/N, where N is the number of points.
            #robust_pixels_k = pixel_weight*2/N
            #robust_pixels_k = 1.5
            robust_pixels_k = self.robust_pixels_k

#            robust_pixels_k = self.robust_pixels_k
#            pixel_weight = 1
#            robust_pixels_k = pixel_weight*2/N

            if robust_pixels_k is not None:
                huber = gtsam.noiseModel.mEstimator.Huber.Create(robust_pixels_k)
                measurement_noise = gtsam.noiseModel.Robust.Create(huber,
                                                                   measurement_noise)

            for j in range(im_pts.shape[1]):
                factor = GenericProjectionFactorCal3_S2(im_pts[:, j],
                                                        measurement_noise,
                                                        X(i), L(point3D_ind[j]),
                                                        self.gtsam_camera)
                wrld_pts_used[point3D_ind[j]] += 1
                self.graph.push_back(factor)

        # Create the data structure to hold the initial estimate to the solution
        # Intentionally initialize the variables off from the ground truth
        print('Setting initial solution')
        self.initial_estimate = Values()

        self.pose_times = image_times
        for i, pose in enumerate(poses):
            self.initial_estimate.insert(X(i), pose)

        self.num_landmarks = len(points3d)
        for j, point in enumerate(points3d):
            self.initial_estimate.insert(L(j), point)

        print('Time elapsed:', time.time() - tic)

    def solve(self):
        tic = time.time()
        # Optimize the graph and print results
        print('Running optimizer')
        if False:
            params = gtsam.DoglegParams()
            params.setAbsoluteErrorTol(1e-6)
            params.setRelativeErrorTol(1e-6)
            params.setVerbosity('TERMINATION')
            optimizer = DoglegOptimizer(self.graph, self.initial_estimate, params)
            print('Optimizing:')
        elif False:
            params = gtsam.GaussNewtonParams()
            params.setAbsoluteErrorTol(1e-6)
            params.setRelativeErrorTol(1e-6)
            params.setVerbosity('TERMINATION')
            optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_estimate, params)
            print('Optimizing:')

            err0 = optimizer.error()
            for _ in range(20):
                result = optimizer.optimize()
                err = optimizer.error()
                print('Relative reduction in error:', (err0-err)/err0)
                #err0 = err
        else:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph,
                                                          self.initial_estimate,
                                                          params)

        self.result = optimizer.optimize()
        print('Time elapsed:', time.time() - tic)
        return optimizer.error()

    def convert_solution(self):
        wrld_pts = np.array([self.result.atPoint3(L(i))
                             for i in range(self.num_landmarks)]).T
        cm2 = copy.deepcopy(self.cm)
        ppp = PlatformPoseInterp(self.cm.platform_pose_provider.lat0,
                                 self.cm.platform_pose_provider.lon0,
                                 self.cm.platform_pose_provider.h0)
        for i in range(len(self.pose_times)):
            t = self.pose_times[i]
            pose = self.result.atPose3(X(i))
            R = pose.rotation().matrix().T
            R = np.dot(self.Rins_to_cam.T, R)
            # The gtsam rotation matrix is a coordinate system rotation.
            quat = quaternion_from_matrix(R.T)
            ppp.add_to_pose_time_series(t, [pose.x(), pose.y(), pose.z()], quat)

        cm2.platform_pose_provider = ppp

        position_err = []
        position_err0 = []
        for i, t in enumerate(self.pose_times):
            pos0, quat0 = self.cm.platform_pose_provider.pose(t)
            pos1 = cm2.platform_pose_provider.pos(t)
            position_err.append(np.linalg.norm(pos0 - pos1))
            position_err0.append(np.linalg.norm(pos0 - self.cm.platform_pose_provider.pos(t)))

        print('Mean position difference reduced from', np.mean(position_err0), 'to', np.mean(position_err))

        return cm2, wrld_pts


