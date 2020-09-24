#! /usr/bin/python
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

"""
from __future__ import division, print_function
import numpy as np
import csv
import copy
import glob
import os
import cv2
import shapely
from shapely.ops import cascaded_union
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from pyembree import rtcore_scene as rtcs
from pyembree.mesh_construction import TriangleMesh


class WorldModel(object):
    def __init__(self):
        pass

    def intersect_rays(self, ray_pos, ray_dir):
        raise NotImplementedError


class WorldModelMesh(WorldModel):
    def __init__(self, mesh_fname):
        mesh = trimesh.load(mesh_fname)
        self.embree_scene = rtcs.EmbreeScene()
        self.vertices = mesh.vertices.astype(np.float32)
        self.faces = mesh.faces
        self.embree_mesh = TriangleMesh(self.embree_scene, self.vertices,
                                        self.faces)

    def intersect_rays(self, ray_pos, ray_dir):
        """Intersect rays with the world model.

        :param ray_pos: Ray starting position(s).
        :type ray_pos: array with shape (3) or (3, N)

        :param ray_dir: Ray direction(s).
        :type ray_dir: array with shape (3) or (3, N)

        """
        res = self.embree_scene.run(ray_pos.astype(np.float32).T,
                                    ray_dir.astype(np.float32).T, output=1)

        ray_inter = res['geomID'] >= 0
        depth = np.zeros(ray_dir.shape[1], dtype=np.float32)
        depth[~ray_inter] = np.nan
        primID = res['primID'][ray_inter]
        u = res['u'][ray_inter]
        v = res['v'][ray_inter]
        w = 1 - u - v

        points = np.zeros((ray_dir.shape[1], 3), dtype=np.float32)
        points[~ray_inter] = np.nan

        vertices = self.vertices
        faces = self.faces

        points[ray_inter] = np.atleast_2d(w).T * vertices[faces[primID][:, 0]] + \
                            np.atleast_2d(u).T * vertices[faces[primID][:, 1]] + \
                            np.atleast_2d(v).T * vertices[faces[primID][:, 2]]


        return points.T
