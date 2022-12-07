from __future__ import division, print_function
import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import glob
import copy
import open3d as o3d
from scipy.interpolate import interp2d, RectBivariateSpline, interpn
from SetCoverPy import setcover
import scipy


# ADAPT imports.
import colmap_processing.camera_models as camera_models
from colmap_processing.world_models import WorldModelMesh
from colmap_processing.colmap_interface import read_images_binary, \
    read_points3D_binary, read_cameras_binary, qvec2rotmat, \
    standard_cameras_from_colmap
import colmap_processing.vtk_util as vtk_util
from colmap_processing.colmap_interface import Image as ColmapImage
from colmap_processing.platform_pose import PlatformPoseInterp
from colmap_processing.rotations import quaternion_from_matrix, \
    quaternion_matrix

%matplotlib auto

# Meshed model generated from 3-D points generated by Colmap.
meshed_model = '/mnt/data10tb/projects/ursa/3d/UAH/model.11-6.ply'

# This should be the result of undistort function.
image_dir = '/mnt/data10tb/projects/ursa/3d/UAH/images'

# This should be the model generated by the undistort function.
sparse_recon_subdir = '/mnt/data10tb/projects/ursa/3d/UAH/sparse'


# --------------------- Read Existing Colmap Reconstruction ------------------
# Read in the Colmap details of all images.
images_bin_fname = '%s/images.bin' % sparse_recon_subdir
colmap_images = read_images_binary(images_bin_fname)
camera_bin_fname = '%s/cameras.bin' % sparse_recon_subdir
colmap_cameras = read_cameras_binary(camera_bin_fname)

sfm_cms = standard_cameras_from_colmap(colmap_cameras, colmap_images)

render_resolution = (1000, 1000)
clipping_range = [.1, 30]

# A ray is launched from each image at the model, and if it intersections
# within this distance from the mesh surface, it is considered a match.
vertex_eps = 1e-2

max_image_width = 1000

# Maximum number of pixels of texture per unit of model length.
max_text_res = 600
# ----------------------------------------------------------------------------

colmap_images_inds = sorted(list(colmap_images.keys()))

if False:
    colmap_images_inds = colmap_images_inds[40:45]

model_reader = vtk_util.load_world_model(meshed_model)
world_mesh = WorldModelMesh(meshed_model)
mesh = o3d.io.read_triangle_mesh(meshed_model, True)
vertices = np.asarray(mesh.vertices)
vertices_colors = np.asarray(mesh.vertex_colors)
triangles = np.asarray(mesh.triangles)

tria_edge_len = []
for i, j, k in triangles:
    tria_edge_len.append([np.sqrt(np.sum((vertices[i] - vertices[j])**2)),
                          np.sqrt(np.sum((vertices[j] - vertices[k])**2)),
                          np.sqrt(np.sum((vertices[i] - vertices[k])**2))])
tria_edge_len = np.array(tria_edge_len)

triangle_covarage = np.zeros((len(triangles), len(colmap_images_inds)),
                             dtype=np.float32)
uv_per_image = {}

# Figure out which images provide the best coverage for each triangle.
for image_id_ind, image_id in enumerate(colmap_images_inds):
    print('Processing \'triangle_covarage\' for image:', image_id)
    image = colmap_images[image_id]

    cm = sfm_cms[image.camera_id]

    # Get depth map.
    P = cm.get_camera_pose(image_id)
    R = P[:3, :3]
    cam_pos = -np.dot(R.T, P[:, 3])

    viz_vert_ind = np.arange(len(vertices))

    im_pts = cm.project(vertices.T, image_id)
    ind = np.all(im_pts >= 0, axis=0)
    ind = np.logical_and(ind, im_pts[0] <= cm.width)
    ind = np.logical_and(ind, im_pts[1] <= cm.height)
    im_pts = im_pts[:, ind]
    viz_vert_ind = viz_vert_ind[ind]

    if im_pts.shape[1] == 0:
        continue

    if False:
        if True:
            ret = vtk_util.render_distored_image(cm.width, cm.height, cm.K,
                                                 cm.dist, cam_pos, R, model_reader,
                                                 return_depth=True,
                                                 monitor_resolution=(1000, 1000),
                                                 clipping_range=[3, 7],
                                                 fill_holes=False)

            #plt.imshow(ret[0])
            #plt.imshow(ret[1])

            if False:
                plt.imshow(ret[0])
                plt.plot(im_pts[0], im_pts[1], 'ro')

            true_depth = ret[1]

            x = np.linspace(0.5, true_depth.shape[1] - 0.5, true_depth.shape[1])
            y = np.linspace(0.5, true_depth.shape[0] - 0.5, true_depth.shape[0])

            if False:
                # Sanity check
                X, Y = np.meshgrid(x, y)
                xi = np.vstack([X.ravel(), Y.ravel()]).T

                z = interpn((x, y), true_depth.T, xi, method='linear',
                            bounds_error=False, fill_value=np.nan)

                Z = np.reshape(z, X.shape)

            d1 = interpn((x, y), true_depth.T, im_pts.T, method='linear',
                        bounds_error=False, fill_value=np.nan)

            # This is the depth that each so-far visible vertex is from the camera.
            d2 = np.dot(R[2], vertices[viz_vert_ind].T - np.atleast_2d(cam_pos).T)

            ind = d1 > d2 - vertex_eps
        else:
            # Determine which vertices are unocculded.
            ray_pos, ray_dir = cm.unproject(im_pts, image_id)

            # Sometimes embree misses vertex.
            d1 = world_mesh.intersect_rays(ray_pos, ray_dir) - np.atleast_2d(cam_pos).T
            d1 = np.sqrt(np.sum(d1**2, axis=0))
            d2 = vertices[viz_vert_ind].T  - np.atleast_2d(cam_pos).T
            d2 = np.sqrt(np.sum(d2**2, axis=0))

            # If it is
            ind = np.abs(d2 - d1) < vertex_eps

        im_pts = im_pts[:, ind]
        viz_vert_ind = viz_vert_ind[ind]

    if False:
        img = cv2.imread('%s/%s' % (image_dir, image.name))[:, :, ::-1].copy()
        s = set(viz_vert_ind)
        for i, j, k in triangles:
            if (i in s) + (j in s) + (k in s) >= 2:
                wrld_pts = np.vstack([vertices[i], vertices[j], vertices[k]]).T
                im_pts = np.round(cm.project(wrld_pts, image_id)).astype(int)
                cv2.drawContours(img, [im_pts.T], 0, 255, thickness=-1)

        plt.imshow(img)

    viz_vert = np.zeros(len(vertices), dtype=bool)
    viz_vert[viz_vert_ind] = True

    inv_map = {viz_vert_ind[i]: i for i in range(len(viz_vert_ind))}

    # These are the triangles that are visible.
    ind = np.where(np.all(viz_vert[triangles], axis=1))[0]

    im_pts = im_pts.T
    for ind_ in ind:
        i, j, k = triangles[ind_]
        i, j, k = inv_map[i], inv_map[j], inv_map[k]

        tria_edge_len_ = np.array([np.sqrt(np.sum((im_pts[i] - im_pts[j])**2)),
                                   np.sqrt(np.sum((im_pts[j] - im_pts[k])**2)),
                                   np.sqrt(np.sum((im_pts[i] - im_pts[k])**2))])
        triangle_covarage[ind_, image_id_ind] = np.min(tria_edge_len_/tria_edge_len[ind_])


# triangle_covarage[i, j] shows the pixel resolution coverage for image j by
# triangle i.


# Any images that are covering a particular triangle with higher resolution
# per unit model length than 'max_text_res', it is unnecassarily high
# resolution, so we clamp to 'max_text_res'.
triangle_covarage = np.minimum(max_text_res, triangle_covarage)

# Remove any images that don't contribute to the mesh.
ind = np.any(triangle_covarage, axis=0)
colmap_images_inds = np.array(colmap_images_inds)[ind]
triangle_covarage = triangle_covarage[:, ind]


if False:
    # Sanity check
    image_id_ind = 65
    image_id = colmap_images_inds[image_id_ind]
    image = colmap_images[image_id]
    img = cv2.imread('%s/%s' % (image_dir, image.name))[:, :, ::-1].copy()

    for tri_ind in np.where(triangle_covarage[:, image_id_ind] > 0)[0]:
        i, j, k = triangles[tri_ind]
        wrld_pts = np.vstack([vertices[i], vertices[j], vertices[k]]).T
        im_pts = np.round(cm.project(wrld_pts, image_id)).astype(int)
        cv2.drawContours(img, [im_pts.T], 0, 255, thickness=-1)

    plt.imshow(img)


# Start by taking the best individual choice.
tri_cov_map = np.argmax(triangle_covarage, axis=1)


# Map length number of triangles indicating which image (Colmap image index)
# will support its text.
if False:
    # If one image covers a triangle at 50% resolution of another image, it is
    # a good enough replacement.
    sufficient_fract = 0.5
    ind = np.max(triangle_covarage, axis=1) > 0
    triangle_covarage_ = triangle_covarage[ind]
    a_matrix = triangle_covarage_ > np.atleast_2d(np.max(triangle_covarage_, axis=1)*sufficient_fract).T

    # Identify any redundant conditions.
    a_matrix = np.array(list(set([tuple(a) for a in a_matrix])))

    g = setcover.SetCover(a_matrix, np.ones(a_matrix.shape[1]), maxiters=1)
    solution, time_used = g.SolveSCP()

    colmap_images_inds = np.array(colmap_images_inds)[g.s]
    triangle_covarage = triangle_covarage[:, g.s]
elif True:
    # Any case where an image provides this fraction of the resolution coverage
    # as the best covering image will be accepted as a subsitute.
    sufficient_fract = 0.75

    # a_matrix[i, j] indicates whether image j is suitable to cover triangle i.
    # The image index j, here, is the index into `colmap_images_inds` to get
    # the actual Colmap image index.
    a_matrix = triangle_covarage > np.atleast_2d(np.max(triangle_covarage, axis=1)*sufficient_fract).T

    # Start by taking the best individual choice.
    tri_cov_map = np.argmax(triangle_covarage, axis=1)

    triangles_by_vertex = {i: [] for i in range(len(vertices))}
    for i, tri in enumerate(triangles):
        for v in tri:
            triangles_by_vertex[v].append(i)

    for i in triangles_by_vertex:
        triangles_by_vertex[i] = list(set(triangles_by_vertex[i]))


    # 'a_matrix' is a boolean array (num_triangles x num_images) where
    # a_matrix[i, j] encodes whether triangles[j] is satisfactorily covered by
    # image colmap_images_inds[j].
    changed = 0
    while True:
        changed0 = changed
        for i in range(len(triangles)):
            tris = set()
            for tri in triangles[i]:
                tris = tris.union(set(triangles_by_vertex[tri]))

            tris.remove(i)

            # 'tris' are the triangle indices of all adjacent triangles.

            neighbor_images = [tri_cov_map[tri] for tri in tris if a_matrix[i, tri_cov_map[tri]]]

            if len(neighbor_images) == 0:
                continue

            best_neighbor = scipy.stats.mode(neighbor_images, keepdims=True)[0][0]
            tri_cov_map[i] = best_neighbor

            changed += 1

        if changed0 == changed:
            break

        print('Changed', changed)
        print('Texture will pull from', len(set(tri_cov_map)), 'images')


if False:
    # Sanity check
    image_id_ind = 0
    image_id = colmap_images_inds[image_id_ind]
    image = colmap_images[image_id]
    img = cv2.imread('%s/%s' % (image_dir, image.name))[:, :, ::-1].copy()

    #plt.plot(np.sort(triangle_covarage[triangle_covarage > 0]))

    # Isolate a triangle.
    for tri_ind in range(len(triangles)):
        if triangle_covarage[tri_ind, image_id_ind] > 0:
            i, j, k = triangles[tri_ind]
            wrld_pts = np.vstack([vertices[i], vertices[j], vertices[k]]).T
            im_pts = np.round(cm.project(wrld_pts, image_id)).astype(int)
            if (np.all(im_pts[0] > 2050) and np.all(im_pts[0] < 2100) and
                np.all(im_pts[1] > 1650) and np.all(im_pts[1] < 1723)):
                print(tri_ind)



    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for tri_ind in range(len(triangles)):
        if triangle_covarage[tri_ind, image_id_ind] > 0:
            v = int(np.round(triangle_covarage[tri_ind, image_id_ind]*255/max_text_res))
            i, j, k = triangles[tri_ind]
            wrld_pts = np.vstack([vertices[i], vertices[j], vertices[k]]).T
            im_pts = np.round(cm.project(wrld_pts, image_id)).astype(int)
            cv2.drawContours(mask, [im_pts.T], 0, color=v, thickness=-1)

    plt.imshow(mask)

    for tri_ind in range(len(triangles)):
        if tri_cov_map[tri_ind] == image_id_ind:
            i, j, k = triangles[tri_ind]
            wrld_pts = np.vstack([vertices[i], vertices[j], vertices[k]]).T
            im_pts = np.round(cm.project(wrld_pts, image_id)).astype(int)
            cv2.drawContours(img, [im_pts.T], 0, 255, thickness=-1)

    plt.imshow(img)


# Create the mask for each image so we know how much to include in the texture.
images_for_texture = {}
for image_id_ind, image_id in enumerate(colmap_images_inds):
    print('Processing texture for image:', image_id)
    image = colmap_images[image_id]

    ind = np.where(tri_cov_map == image_id_ind)[0]

    if len(ind) == 0:
        continue

    img = cv2.imread('%s/%s' % (image_dir, image.name))[:, :, ::-1].copy()

    #raise Exception()

    cm = sfm_cms[image.camera_id]

    if False:
        # Draw triangles.
        image_mask = np.zeros((cm.height, cm.width), dtype=np.uint8)

        for i, j, k in triangles[ind]:
            wrld_pts = np.vstack([vertices[i], vertices[j], vertices[k]]).T
            im_pts = np.round(cm.project(wrld_pts, image_id)).astype(int)
            cv2.drawContours(image_mask, [im_pts.T], 0, 255, thickness=4)

        indh = np.where(np.any(image_mask, axis=0))[0]
        indv = np.where(np.any(image_mask, axis=1))[0]

        # left, right, top, bottom
        left, right, top, bottom = indh[0], indh[-1], indv[0], indv[-1]
    else:
        vertices[list(set(triangles[ind].ravel()))]
        im_pts = cm.project(vertices[list(set(triangles[ind].ravel()))].T, image_id)
        left = int(np.floor(min(im_pts[0])))
        right = int(np.ceil(max(im_pts[0])))
        top = int(np.floor(min(im_pts[1])))
        bottom = int(np.ceil(max(im_pts[1])))

    left -= 2
    right += 2
    top -= 2
    bottom += 2

    if True:
        # Use entire image.
        left = 0
        right = cm.width
        top = 0
        bottom = cm.height

    left = max([left, 0])
    top = max([top, 0])
    right = min([right, cm.width])
    bottom = min([bottom, cm.height])

    in_w, in_h = right - left, bottom - top

    s = max_image_width/in_w
    s = min([s, 1])
    final_height = int(np.ceil(in_h*s))
    final_width = max_image_width

    x = (right + left)/2.0
    y = (top + bottom)/2.0
    M = np.array([[1/s, 0, x - final_width/2/s],
                  [0, 1/s, y - final_height/2/s],
                  [0, 0, 1]])

    if s < 0.75:
        flags = cv2.INTER_AREA | cv2.WARP_INVERSE_MAP
    else:
        flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP

    warped_img = cv2.warpAffine(img, M[:2], (final_width, final_height),
                                borderValue=(0, 0, 0), flags=flags)

    images_for_texture[image_id] = [np.linalg.inv(M), warped_img]


if True:
    # Only works when all images are the same resolution
    shape0 = None
    for image_id in images_for_texture:
        if shape0 is None:
            shape0 = images_for_texture[image_id][1].shape
        else:
            assert images_for_texture[image_id][1].shape == shape0

    h, w = shape0[:2]
    L = len(images_for_texture)
    #L = 310
    if L == 1:
        nrows = 1
        ncols = 1
    else:
        # Do something silly and inefficient for now.
        best_nrows = None
        best_ncols = None
        min_max_res = np.inf
        for nrows in range(1, L):
            for ncols in range(1, L):
                if nrows*ncols < L:
                    continue

                curr = max(nrows*h, ncols*w)
                if curr < min_max_res:
                    min_max_res = curr
                    best_nrows = nrows
                    best_ncols = ncols

        nrows = best_nrows
        ncols = best_ncols

    offsets = []
    for i in range(nrows):
        for j in range(ncols):
            offsets.append([i*h, j*w])

    texture = np.zeros((int(nrows*h), int(ncols*w), 3), dtype=np.uint8)

    # Make the texture
    final_homographies = {}
    for i, image_id in enumerate(images_for_texture):
        #print('Processing image:', image_id)
        # M is a homography that takes raw-image coordinates and returns the
        # reduced-resolution image coordinates.
        M, img = images_for_texture[image_id]

        dy, dx = offsets[i]

        M = np.dot([[1, 0, dx], [0, 1, dy], [0, 0, 1]], M)[:2]

        texture[dy:dy+img.shape[0], dx:dx+img.shape[1], :] = img
        final_homographies[image_id] = M


# v_uv is a 3*num_triangles x 2 array providing normalized texture coordinates
# for each triangle.
v_uv = np.zeros((3*len(triangles), 2), dtype=np.float32)
for image_id_ind, image_id in enumerate(colmap_images_inds):
    print('Generating UV for Image:', image_id)
    try:
        M = final_homographies[image_id]
    except KeyError:
        continue

    image = colmap_images[image_id]
    cm = sfm_cms[image.camera_id]

    if False:
        # Sanity check
        image = colmap_images[image_id]
        img = cv2.imread('%s/%s' % (image_dir, image.name))[:, :, ::-1].copy()
        plt.imshow(img)

    # Indices of the triangles we care about.
    ind = np.where(tri_cov_map == image_id_ind)[0]

    for ind_ in ind:
        i, j, k = triangles[ind_]
        wrld_pts = np.vstack([vertices[i], vertices[j], vertices[k]]).T
        im_pts = cm.project(wrld_pts, image_id)

        if (np.any(im_pts < 0) or np.any(im_pts[0] > cm.width)
            or np.any(im_pts[1] > cm.height)):
            continue

        im_pts = np.vstack([im_pts, [1, 1, 1]])
        im_pts = np.dot(M, im_pts)

        v_uv[ind_*3, 0] = im_pts[0, 0]/texture.shape[1]
        v_uv[ind_*3 + 1, 0] = im_pts[0, 1]/texture.shape[1]
        v_uv[ind_*3 + 2, 0] = im_pts[0, 2]/texture.shape[1]

        v_uv[ind_*3, 1] = im_pts[1, 0]/texture.shape[0]
        v_uv[ind_*3 + 1, 1] = im_pts[1, 1]/texture.shape[0]
        v_uv[ind_*3 + 2, 1] = im_pts[1, 2]/texture.shape[0]


mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
mesh.textures=[o3d.geometry.Image(texture)]

#o3d.visualization.draw_geometries([mesh])

fname = '/mnt/data10tb/projects/ursa/3d/UAH/test.obj'
print('Saving: \'%s\'' % fname)
o3d.io.write_triangle_mesh(fname, mesh, write_vertex_colors=False,
                           write_triangle_uvs=True, compressed=True)
