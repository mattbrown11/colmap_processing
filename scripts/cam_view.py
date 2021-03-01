#!/usr/bin/env python
# ckwg +31
# Copyright 2021 by Kitware, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
Eugene.Borovikov@Kitware.com: render colmap camera view given a 3D model in PLY format.
'''
import argparse, logging, cv2, numpy as np
import matplotlib.pyplot as plt
from colmap_processing.geo_conversions import llh_to_enu
from colmap_processing.static_camera_model import load_static_camera_from_file
import colmap_processing.vtk_util as vtk_util

def run(args):
    save_dir = args.output_path
    mesh_lat0, mesh_lon0, mesh_h0 = args.LLH0
    # VTK renderings are limited to monitor resolution (width x height).
    monitor_resolution = (1920, 1080) # TODO: param/config
    model_reader = vtk_util.load_world_model(args.input_mesh)
    height, width, K, dist, R, depth_map, latitude, longitude, altitude = load_static_camera_from_file(args.input_cam)
    cam_pos = llh_to_enu(latitude, longitude, altitude, mesh_lat0, mesh_lon0, mesh_h0)
### render camera view
    render_resolution = list(monitor_resolution)
    vfov = np.arctan(render_resolution[1]/2/K[1,1])*360/np.pi
    vtk_camera = vtk_util.Camera(render_resolution[0], render_resolution[1], vfov, cam_pos, R)
    img = vtk_camera.render_image(model_reader, clipping_range=[1, 2000],
                                diffuse=0.6, ambient=0.6, specular=0.1,
                                light_color=[1.0, 1.0, 1.0],
                                light_pos=[0, 0, 1000])
# from colmap_processing.vtk_util import render_distored_image
#     img = render_distored_image(width, height, K, dist, cam_pos, R,
#                                       model_reader, return_depth=True,
#                                       monitor_resolution=monitor_resolution,
#                                       clipping_range=[1, 2000])[0]
    plt.imshow(img)
    plt.show()
    cv2.imwrite('{}/cam.view.png'.format(save_dir), img[:,:,::-1])

def CLI(argv=None):
    LLH0NSR = np.array([42.43722062, -84.02781521, 251.412]) # North Star Reach origin in LLH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i:c', '--input_cam', metavar='path', default='../data/NorthStarReach/202010/camera_models/axisptz2/ref_view.cam.yml',
                        help='path/to/input/camera/model.yml; default=%(default)s')
    parser.add_argument('-i:m', '--input_mesh', metavar='path', default='../data/NorthStarReach/202010/SLAM/georeg/poisson.ply',
                        help='path/to/input/mesh.ply; default=%(default)s')
    parser.add_argument('-l', '--log', metavar='level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        default='WARNING', help='logging verbosity level: %(choices)s; default=%(default)s')
    parser.add_argument('-o', '--LLH0', metavar='value', default=LLH0NSR,
                        help='assumed local GPS origin as an explicit vector [latitude,longitude,altitude]; default=%(default)s')
    parser.add_argument('-o:p', '--output_path', metavar='path', default='../data/NorthStarReach/202010/camera_models/axisptz2',
                        help='path/to/output/model/folder; default=%(default)s')
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log)
    run(args)

if __name__ == '__main__': CLI()
