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
Eugene.Borovikov@Kitware.com: load and print camera parameters from YAML, and optionally render the view given a 3D model in PLY format.
'''
import logging, sys, numpy as np
import matplotlib.pyplot as plt

def run(args):
### load camera
    CamMdlFN = args.input_cam
    from colmap_processing.static_camera_model import load_static_camera_from_file
    height, width, K, dist, R, depth_map, latitude, longitude, altitude = load_static_camera_from_file(CamMdlFN)
### print camera parameters
    print(CamMdlFN)
    print('W,H={}'.format([width, height]))
    np.savetxt(sys.stdout, K, '%g', '\t', header='K')
    vFoV = np.arctan2(K[1,2], K[1,1])*180/np.pi
    print('vFoV={}'.format(vFoV))
    np.savetxt(sys.stdout, R, '%g', '\t', header='R')
    print('LLH={}'.format([latitude, longitude, altitude]))
    mesh_lat0, mesh_lon0, mesh_h0 = args.LLH0
    from colmap_processing.geo_conversions import llh_to_enu
    cam_pos = llh_to_enu(latitude, longitude, altitude, mesh_lat0, mesh_lon0, mesh_h0)
    print('cam_pos={}'.format(cam_pos))
    CCCP = np.eye(4)
    CCCP[:3,:4] = np.array([R[0],-R[1],-R[2], cam_pos]).T
    np.savetxt(sys.stdout, CCCP, '%g', '\t', header='CCCP')
    if not args.visual: return 0
### VTK rendering: limited to monitor resolution (width x height)
    monitor_resolution = (1920, 1080) # TODO: param/config
    import colmap_processing.vtk_util as vtk_util
    model_reader = vtk_util.load_world_model(args.input_mesh)
### render camera view
    render_resolution = list(monitor_resolution)
    vfov = np.arctan(render_resolution[1]/K[1,1])*180/np.pi
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
### save camera view to an image file
    import os
    CamViewFN = os.path.join(os.path.dirname(CamMdlFN), 'vtk_view.png')
    import cv2 as cv
    cv.imwrite(CamViewFN, img[:,:,::-1])

def CLI(argv=None):
    import argparse
    CamName = 'axisptz9'
    CamDir = 'axisptz6_7_9'
    CamModelFN = '../data/NorthStarReach/202102/calibration/{}/camera_models/{}/camera_model.yaml'.format(CamDir, CamName) 
    LLH0NSR = np.array([42.43722062, -84.02781521, 251.412]) # North Star Reach origin in LLH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i:c', '--input_cam', metavar='path', default=CamModelFN,
                        help='path/to/input/camera/model.yml; default=%(default)s')
    parser.add_argument('-i:m', '--input_mesh', metavar='path', default='../data/NorthStarReach/202010/archive/202011/mesh.geo.ply',
                        help='path/to/input/mesh.ply; default=%(default)s')
    parser.add_argument('-l', '--log', metavar='level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        default='WARNING', help='logging verbosity level: %(choices)s; default=%(default)s')
    parser.add_argument('-o', '--LLH0', metavar='value', default=LLH0NSR,
                        help='assumed local GPS origin as an explicit vector [latitude,longitude,altitude]; default=%(default)s')
    parser.add_argument('-v', '--visual', action='store_true', help='display/save visuals')
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log)
    run(args)

if __name__ == '__main__': CLI()
