#!/usr/bin/env python
# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
'''
Eugene.Borovikov@kitware.com: model visualization adapted from a COLMAP provided example.
Use typical 3D mouse manipulation gestures. ESC to exit.
'''

import argparse, logging, open3d
import numpy as np

from colmap_processing.colmap_interface import read_model, qvec2rotmat


class Model: # COLMAP model visualization facility
    def __init__(self, path=None, ext=None):
        self.cameras = []
        self.images = []
        self.points3D = []
        self.__vis = None
        if not path: return 
        self.read_model(path, ext)
        self.create_window()
        self.add_points()
        self.add_cameras(scale=0.25)


    def read_model(self, path, ext=None):
        self.cameras, self.images, self.points3D = read_model(path, ext)

    def add_points(self, min_track_len=3, remove_statistical_outlier=True):
        pcd = open3d.geometry.PointCloud()
        xyz = []
        rgb = []
        for point3D in self.points3D.values():
            track_len = len(point3D.point2D_idxs)
            if track_len < min_track_len: continue
            xyz.append(point3D.xyz)
            rgb.append(point3D.rgb / 255)
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)
        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    def add_cameras(self, scale=1):
        frames = []
        for img in self.images.values():
            # rotation
            R = qvec2rotmat(img.qvec)
            # translation
            t = img.tvec
            # pose
            t = -R.T.dot(t)
            R = R.T
            # intrinsics
            cam = self.cameras[img.camera_id]
            if cam.model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
                fx = fy = cam.params[0]
                cx = cam.params[1]
                cy = cam.params[2]
            elif cam.model in ('PINHOLE', 'OPENCV', 'OPENCV_FISHEYE'):
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
            else:
                raise Exception('unsupported camera model: {}'.format(cam.model))
            K = np.identity(3)
            K[0,0] = fx
            K[1,1] = fy
            K[0,2] = cx
            K[1,2] = cy
            # create axis, plane and pyramid geometries that will be drawn
            cam_model = draw_camera(K, R, t, cam.width, cam.height, scale)
            frames.extend(cam_model)
        # add geometries to visualizer
        for i in frames:
            self.__vis.add_geometry(i)

    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()
        opt = self.__vis.get_render_option()
        opt.background_color = np.asarray([.9, .9, .9])
        opt.point_size = 1.0

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()


def draw_camera(K, R, t, w, h, scale=1, color=[0.8, 0.2, 0.8]):
    '''Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    '''
    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)
    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))
    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    axis.transform(T)
    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]
    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]
    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)
    # pyramid
    points_in_world = [(R.dot(p) + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for _ in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)
    # return as list in Open3D format
    return [axis, plane, line_set]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_model', help='path/to/input/model/folder')
    parser.add_argument('--input_format', choices=['.bin', '.txt'], default='.bin', help='input model format: %(choices)s; default=%(default)s')
    parser.add_argument('-l', '--log', metavar='level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        default='WARNING', help='logging verbosity level: %(choices)s; default=%(default)s')
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log)
    return args


def CLI(argv=None):
    args = parse_args(argv)
    model = Model(args.input_model, ext=args.input_format)
    logging.info('cameras: {}'.format(len(model.cameras)))
    logging.info('images: {}'.format(len(model.images)))
    logging.info('points3D: {}'.format(len(model.points3D)))
### display using Open3D visualization tools
    model.show()


if __name__ == '__main__': CLI()
