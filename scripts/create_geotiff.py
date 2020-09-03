#! /usr/bin/python
"""
ckwg +31
Copyright 2018 by Kitware, Inc.
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

"""
from __future__ import division, print_function
import numpy as np
import os
import cv2
import subprocess
import matplotlib.pyplot as plt
import glob
import trimesh
import math
import PIL
from osgeo import osr, gdal

import colmap_processing.vtk_util as vtk_util
from colmap_processing.geo_conversions import enu_to_llh


# ----------------------------------------------------------------------------
if True:
    mesh_fname = 'everything.ply'
    save_dir = 'meshed'

    # Latitude and longitude associated with (0, 0, 0) in the model.
    latitude0 = 0    # degrees
    longitude0 = 0  # degrees
    altitude0 = 0  # degrees

    # Set GSD of orthographic base layer.
    gsd = 0.05  # meters


# VTK renderings are limited to monitor resolution (width x height).
monitor_resolution = (1920, 1080)
# ----------------------------------------------------------------------------


# Determine the bounds of the model.
mesh = trimesh.load(mesh_fname)
pts = mesh.vertices.T
model_bounds = np.array([[min(pts[i]), max(pts[i])] for i in range(3)])

if False:
    w = np.diff(model_bounds[0])
    h = np.diff(model_bounds[1])
    w = 400
    h = 450
    c = np.sum(np.array(model_bounds[:2]), 1)/2
    model_bounds = np.array([[c[0] - w/2, c[0] + w/2],
                             [c[1] - h/2, c[1] + h/2],
                             model_bounds[2]])
    model_bounds = np.array([[-220, 180], [-240, 210], model_bounds[2]])

delta_xyz = np.diff(model_bounds)

# Read model into VTK.
model_reader = vtk_util.load_world_model(mesh_fname)


def get_ortho_image(xbnds, ybnds, res_x, res_y):
    # Model orthographic as camera that is 'alt' meters high.
    alt = 1e4*float(delta_xyz[2]) + model_bounds[2, 1]
    #alt = 1e3

    # Position of the camera.
    pos = [np.mean(xbnds), np.mean(ybnds), model_bounds[2, 1] + alt]

    assert res_x < monitor_resolution[0]
    assert res_y < monitor_resolution[1]

    dy = float(np.abs(np.diff(ybnds)))
    vfov = 2*np.arctan(dy/2/alt)*180/np.pi
    ortho_camera = vtk_util.CameraPanTilt(res_x, res_y, vfov, pos, 0, -90)

    clipping_range = [alt, alt + float(delta_xyz[2])]
    #clipping_range = [1e3, 2e4]
    img = ortho_camera.render_image(model_reader,
                                    clipping_range=clipping_range,
                                    diffuse=0.6, ambient=0.6, specular=0.1,
                                    light_color=[1.0, 1.0, 1.0],
                                    light_pos=[0,0,1000])

    ret = ortho_camera.unproject_view(model_reader,
                                      clipping_range=clipping_range)
    z = ret[2] + altitude0
    return img, z


# Build up geotiff by tiling the rendering.
full_res_x = int(math.ceil(delta_xyz[0]/gsd))
full_res_y = int(math.ceil(delta_xyz[1]/gsd))
ortho_image = np.zeros((full_res_y, full_res_x, 3), dtype=np.uint8)
dem = np.zeros((full_res_y, full_res_x), dtype=np.float)

num_cols = int(math.ceil(full_res_x/monitor_resolution[0]))
num_rows = int(math.ceil(full_res_y/monitor_resolution[1]))
indr = np.round(np.linspace(0, full_res_y, num_rows + 1)).astype(np.int)
indc = np.round(np.linspace(0, full_res_x, num_cols + 1)).astype(np.int)
xbnds_array = (indc/full_res_x)*delta_xyz[0] + model_bounds[0, 0]
ybnds_array = model_bounds[1, 1] - (indr/full_res_y)*delta_xyz[1]

for i in range(num_rows):
    for j in range(num_cols):
        ret = get_ortho_image(xbnds_array[j:j+2],
                              ybnds_array[i:i+2],
                              int(indc[j+1] - indc[j]),
                              int(indr[i+1] - indr[i]))
        ortho_image[indr[i]:indr[i+1], indc[j]:indc[j+1], :] = ret[0]
        dem[indr[i]:indr[i+1], indc[j]:indc[j+1]] = ret[1]


print('DEM elevation ranges from %0.4f to %0.4f' %
      (dem.ravel().min(), dem.ravel().max()))


# ----------------------------------------------------------------------------
# Identify holes in the model and then inpaint them.
hole_mask = dem < model_bounds[2, 0] - 0.1

output = cv2.connectedComponentsWithStats(hole_mask.astype(np.uint8), 8, cv2.CV_32S)
num_labels = output[0]
labels = output[1]
stats = output[2]
centroids = output[3]

# Remove components that touch outer boundary.
edge_labels = set(labels[:, 0])
edge_labels = edge_labels.union(set(labels[0, :]))
edge_labels = edge_labels.union(set(labels[:, -1]))
edge_labels = edge_labels.union(set(labels[-1, :]))

for i in edge_labels:
    labels[labels == i] = 0

mask = (labels > 0).astype(np.uint8)
ortho_image = cv2.inpaint(ortho_image, mask, 3, cv2.INPAINT_NS)
dem = cv2.inpaint(dem.astype(np.float32), mask, 3, cv2.INPAINT_NS)

im = PIL.Image.fromarray(dem.astype(np.float32), mode='F') # float32
depth_map_fname = '%s/base_layer.dem.tif' % save_dir
im.save(depth_map_fname)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Save GeoTIFF
geotiff_fname = '%s/base_layer.tif' % save_dir
gdal_drv = gdal.GetDriverByName('GTiff')
wgs84_cs = osr.SpatialReference()
wgs84_cs.SetWellKnownGeogCS("WGS84")
wgs84_wkt = wgs84_cs.ExportToPrettyWkt()
gdal_settings = ['COMPRESS=JPEG', 'JPEG_QUALITY=%i' % 90]
ds = gdal_drv.Create(geotiff_fname, ortho_image.shape[1], ortho_image.shape[0],
                     ortho_image.ndim, gdal.GDT_Byte, gdal_settings)
ds.SetProjection(wgs84_cs.ExportToWkt())

enus = [[model_bounds[0, 0], model_bounds[1, 1], 0],
        [model_bounds[0, 1], model_bounds[1, 0], 0]]
lat_lon = [enu_to_llh(enu[0], enu[1], enu[2], latitude0, longitude0, 0)
           for enu in enus]
lat_lon = np.array(lat_lon)[:, :2]

A = np.zeros((2, 3))
A[:, 2] = lat_lon[0][::-1]

dll = lat_lon[1] - lat_lon[0]
A[0, 0] = dll[1] / full_res_x
A[1, 1] = dll[0] / full_res_y

# Xp = padfTransform[0] + P*padfTransform[1] + L*padfTransform[2]
# Yp = padfTransform[3] + P*padfTransform[4] + L*padfTransform[5]
geotrans = [A[0, 2], A[0, 0], A[0, 1], A[1, 2], A[1, 0], A[1, 1]]
ds.SetGeoTransform(geotrans)

if ds.RasterCount == 1:
    ds.GetRasterBand(1).WriteArray(ortho_image[:, :], 0, 0)
else:
    for i in range(ds.RasterCount):
        ds.GetRasterBand(i+1).WriteArray(ortho_image[:, :, i], 0, 0)

ds.FlushCache()  # Write to disk.
ds = None
# ----------------------------------------------------------------------------