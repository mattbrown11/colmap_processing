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
Eugene_Borovikov@Kitware.com: read COLMAP model files and output camera models in YAML.
'''
from __future__ import print_function
import argparse, os, sys, logging, numpy as np
import sensor_models.camera_models as camera_models
from sensor_models.nav_conversions import enu_to_llh
from colmap_processing.colmap_interface import read_model, qvec2rotmat
from sklearn.preprocessing import PolynomialFeatures
from georegister_3Dmodel import hmgRt

def read(path, ext='.bin', lst=None):
    cameras, images, points = read_model(path, ext)
    if not lst is None:
        with open(lst, 'r') as f:
            lst = {e.strip() for e in list(f)}
        images = {k:v for k,v in images.iteritems() if v.name in lst}
        lst = set(images.keys())
        points = {k:v for k,v in points.iteritems() if set(v.image_ids)&lst}
        lst = {v.camera_id for v in images.values()}
        cameras = {k:v for k,v in cameras.iteritems() if k in lst}
    return cameras, images, points

def cam_intrinsics(cam):
    K = np.eye(3)
    # focal length
    K[0,0]=K[1,1]=cam.params[0]
    # principal point
    K[0,2]=cam.params[1]
    K[1,2]=cam.params[2]
    # distortion
    dist = [cam.params[3],0,0,0]
    return K, dist

def printMx(title, M, file=sys.stdout):
    print(title, file=file)
    np.savetxt(file, M, delimiter='\t', fmt='%.8f')

def write_camera_model(cam, img, pts, path, ext='.yml', LLH0=[0,0,0], depth_poly_power=1): # output image camera model and depth map
    BN = os.path.splitext(img.name)[0]
    K, dist = cam_intrinsics(cam) # camera intrinsic matrix and distortion coefficients
    printMx('{} K:'.format(BN), K)
    C = hmgRt(K, np.zeros(3))
    printMx('{} C:'.format(BN), C)
    R = qvec2rotmat(img.qvec) # camera rotation matrix: from colmap to camera coordinates
    cam_pos = -R.T.dot(img.tvec) # camera position in colmap
    P = hmgRt(R, img.tvec)
    printMx('{} P:'.format(BN), P)
    M = C.dot(P)
    printMx('{} M:'.format(BN), M)
### depth map
    logging.info('computing depth map')
    p3Dndx = img.point3D_ids>0
    p3Ds = img.point3D_ids[p3Dndx]
    wrld_pts = np.array([pts[i].xyz for i in p3Ds])
    xyz = wrld_pts-cam_pos
    optical_axis = R[2,:]
    optical_axis /= np.linalg.norm(optical_axis)
    depth = np.dot(xyz, optical_axis)
    poly = PolynomialFeatures(depth_poly_power)
    im_pts = img.xys[p3Dndx]
    feat = poly.fit_transform(im_pts)
    c = np.linalg.lstsq(feat, depth)[0]
    X, Y = np.meshgrid(np.linspace(0.5, cam.width-1, cam.width), np.linspace(0.5, cam.height-1, cam.height))
    im_pts_all = np.vstack([X.ravel(), Y.ravel()]).T
    depth_map = np.dot(poly.fit_transform(im_pts_all), c)
    depth_map.shape = (cam.height, cam.width)
### camera model
    lat0, lng0, alt0 = LLH0 # local geo-origin
    lat, lng, alt = enu_to_llh(cam_pos[0], cam_pos[1], cam_pos[2], lat0, lng0, alt0) # camera geo-coordinates
    DN = os.path.dirname(img.name)
    image_topic = frame_id = os.path.basename(DN)
    camera_model = camera_models.StaticCamera(cam.width, cam.height, K, dist, depth_map,
                                              lat, lng, alt, R, image_topic, frame_id)
    FN = os.path.join(path, BN+'.cam'+ext)
    logging.info('output camera model {}'.format(FN))
    camera_model.save_to_file(FN)
    return camera_model
    
def write(cameras, images, points, path, ext='.yml', LLH0=[0,0,0], depth_poly_power=1):
    for img in images.values():
        write_camera_model(cameras[img.camera_id], img, points, path, ext, LLH0, depth_poly_power)

def arrayArg(arg): # parse array from command line argument
    return np.fromstring(arg.strip('[').strip(']'), dtype=float, sep=',') if type(arg)==str else arg

def run(args):
    cameras, images, points = read(args.input_path, args.input_ext, args.image_list) 
    if args.items in ['cameras', 'all']:
        print('cameras: {}'.format(len(cameras)))
        logging.debug('cameras: {}'.format(cameras))
    if args.items in ['images', 'all']:
        print('images: {}'.format(len(images)))
        logging.debug('images: {}'.format(images))
    if args.items in ['points', 'all']:
        print('points: {}'.format(len(points)))
        logging.debug('points: {}'.format(points))
    if args.output_path is None: return
    write(cameras, images, points, args.output_path, args.output_ext, arrayArg(args.LLH0), args.depth_poly_power)

def CLI(argv=None):
    LLH0NSR = np.array([42.43722062, -84.02781521, 251.412]) # North Star Reach origin in LLH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d:p', '--depth_poly_power', metavar='exp', type=int, default=3, help='depth polynomial power exponent, higher=slower; default=%(default)s')
    parser.add_argument('-i', '--items', metavar='name', choices=['cameras', 'images', 'points', 'all'],
                        default='all', help='items to print/log: %(choices)s; default=%(default)s')
    parser.add_argument('-i:l', '--image_list', metavar='path', default='../data/NorthStarReach/202010/SLAM/lists/cam.lst',
                        help='path/to/image.lst to restrict calibration only to those; default=%(default)s')
    parser.add_argument('-i:p', '--input_path', metavar='path', default='../data/NorthStarReach/202010/SLAM/sparse/1cg',
                        help='path/to/colmap/model/folder; default=%(default)s')
    parser.add_argument('-i:x', '--input_ext', metavar='ext', choices=['.bin','.txt'], default='.bin', help='input model format; default=%(default)s')
    parser.add_argument('-o', '--LLH0', metavar='value', default=LLH0NSR,
                        help='assumed local GPS origin as an explicit vector [latitude,longitude,altitude]; default=%(default)s')
    parser.add_argument('-l', '--log', metavar='level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        default='WARNING', help='logging verbosity level: %(choices)s; default=%(default)s')
    parser.add_argument('-o:p', '--output_path', metavar='path', default='../data/NorthStarReach/202010',
                        help='path/to/output/folder; default=%(default)s')
    parser.add_argument('-o:x', '--output_ext', metavar='ext', default='.yml', help='output model format; default=%(default)s')
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log)
    run(args)

if __name__ == '__main__': CLI()
