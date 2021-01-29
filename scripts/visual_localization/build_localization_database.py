#!/usr/bin/env python
"""
ckwg +31
Copyright 2019 by Kitware, Inc.
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
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import time
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import cv2

# Colmap Processing imports.
from colmap_processing.geo_conversions import llh_to_enu
from colmap_processing.colmap_interface import read_images_binary, Image, \
    read_points3d_binary, read_cameras_binary, qvec2rotmat, \
    standard_cameras_from_colmap
from colmap_processing.database import COLMAPDatabase, pair_id_to_image_ids, \
    blob_to_array
from colmap_processing.vtk_util import render_distored_image
import colmap_processing.vtk_util as vtk_util


# ----------------------------------------------------------------------------
# Base path to the colmap directory.
colmap_data_dir = '/media/mattb/7e7167ba-ad6f-4720-9ced-b699f49ba3aa/ursa/20201012_north_star_reach/building_cal2'

# Image directory.
image_dir = '%s/images0' % colmap_data_dir

# Path to the images.bin file.
images_bin_fname = '%s/images.bin' % colmap_data_dir

# Path to the images.bin file.
camera_bin_fname = '%s/cameras.bin' % colmap_data_dir

# Path to the points3D.bin file.
points_3d_bin_fname = '%s/points3D.bin' % colmap_data_dir

# Meshed model location.
meshed_model_fname = '%s/mesh_smaller.ply' % colmap_data_dir

# Test image not from original set.
test_image_fname = '/media/mattb/7e7167ba-ad6f-4720-9ced-b699f49ba3aa/ursa/20201012_north_star_reach/building_cal1/images/IMG_0795.JPG'

# Monitor resolution for rendering depth maps.
monitor_resolution = (1920, 1080)
# ----------------------------------------------------------------------------


# Read model into VTK.
try:
    # Avoid having to re-read when running interactively.
    model_reader
    assert prev_loaded_fname == meshed_model_fname
except:
    model_reader = vtk_util.load_world_model(meshed_model_fname)
    prev_loaded_fname = meshed_model_fname

# Read in the details of all images.
colmap_images = read_images_binary(images_bin_fname)

# Read cameras.
cameras = read_cameras_binary(camera_bin_fname)

# Create camera models.
std_cams = standard_cameras_from_colmap(cameras, colmap_images)


# Define a convenience object that manages all projection operations from image.
class Image(object):
    def __init__(self, colmap_image, std_cams, model_reader, keypoints=None,
                 descriptors=None):

        self.camera_model = std_cams[colmap_image.camera_id]
        self.image_filename = '%s/%s' % (image_dir, colmap_image.name)
        self.image_id = colmap_image.id
        self.img_time = colmap_image.id
        self._image = None
        self.model_reader = model_reader
        self.keypoints = keypoints
        self.descriptors = descriptors

    @property
    def image(self):
        if self._image is None:
            image = cv2.imread(self.image_filename)

            if image.ndim == 3:

                image = image[:, :, ::-1]
            self._image = image

        return self._image

    def render_model(self, clipping_range=[0.01, 20000]):
        """Return image rendered frm the 3-D model.

        """
        # 'platform_pose_provider' stored with the camera model has been
        # overloaded to treat 'image_num' as time. Therefore, passing
        # t = image_num will recover the pose of the camera when it took image
        # 'image_num'.
        camera = self.camera_model
        pose_mat = camera.get_camera_pose(self.img_time)
        R = pose_mat[:3, :3]
        cam_pos = -np.dot(R.T, pose_mat[:,3])

        img2 = render_distored_image(camera.width, camera.height, camera.K,
                                     camera.dist, cam_pos, R, model_reader,
                                     return_depth=False,
                                     monitor_resolution=monitor_resolution,
                                     clipping_range=clipping_range,
                                     fill_holes=False)

        return img2

    def project(self, points):
        """Project world points into the image at a particular time.

        :param points: Coordinates of a point or points within the world
            coordinate system. The coordinate may be Cartesian or homogenous.
        :type points: array with shape (3), (4), (3,n), (4,n)

        :param t: Time at which to project the point(s) (time in seconds since
            Unix epoch).
        :type t: float

        :return: Image coordinates associated with points.
        :rtype: numpy.ndarray of size (2,n)

        """
        return self.camera_model.project(points, t=self.img_time)

    def unproject(self, points, normalize_ray_dir=True):
        """Unproject image points into the world at a particular time.

        :param points: Coordinates of a point or points within the image
            coordinate system. The coordinate may be Cartesian or homogenous.
        :type points: array with shape (2), (2,N), (3) or (3,N)

        :param t: Time at which to unproject the point(s) (time in seconds
            since Unix epoch).
        :type t: float

        :param normalize_dir: If set to True, the ray directions will be
            normalized to a unit a lenght. If set to False, the projection of
            the ray direction vectors onto the optical axis of the camera
            (the z-axis of the camera coordinate system) will have unit
            magnitude, useful when a depth-map projection distance is to be
            applied.
        :type normalize_dir: bool

        :return: Ray position and direction corresponding to provided image
            points. The direction points from the center of projection to the
            points. The direction vectors are not necassarily normalized.
        :rtype: [ray_pos, ray_dir] where both of type numpy.ndarry with shape
            (3,n).

        """
        return self.camera_model.unproject(points, t=self.img_time,
                                           normalize_ray_dir=normalize_ray_dir)


images = {}
for image_num in colmap_images:
    # Look at the Colmap files.
    colmap_image = colmap_images[image_num]

    images[image_num] = Image(colmap_image, std_cams, model_reader)


if False:
    # Sanity check that the views agree with the 3-D model.
    image_num = 90
    plt.close('all')
    plt.figure();   plt.imshow(images[image_num].render_model())
    plt.figure();   plt.imshow(images[image_num].image)


class image_feature(object):
    __slots__ = ['im_pt', 'image', 'proj_matrix', 'descriptor', 'matches',
                 'view_pos', 'angular_size', 'size_pixels', 'xyz']

    def __init__(self, im_pt, image, descriptor, proj_matrix, view_pos,
                 size_pixels, angular_size):
        """

        :param proj_matrix: Virtual camera matrix that maps the world feature to
            unit circle centered at (0, 0) and with its major axis aligned
            with positive x direction in a virtual image coordinate system.
        :type proj_matrix: 3x4 matrix

        """
        self.im_pt = im_pt
        self.image = image
        self.proj_matrix = proj_matrix
        self.view_pos = view_pos
        self.size_pixels = size_pixels
        self.angular_size = angular_size
        self.descriptor = descriptor
        self.matches = []
        self.xyz = None


orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=20,
                     edgeThreshold=31, firstLevel=0, patchSize=31)


image_features = {}
for image_num in images:
    print('Calculating ORB features for image % i' % image_num)
    image = images[image_num]
    kp, des = orb.detectAndCompute(image.image, None)

    # Remove the cached image so we don't run out of memory.
    image._image = None

    if False:
        plt.imshow(cv2.drawKeypoints(img, kp, img.copy(), color=(255,0,0),
                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    image.keypoints = kp
    image.descriptors = des

    image_features_ = []
    for i in range(len(kp)):
        # Launch a ray from the feature's center.
        cam_pos, zdir = image.unproject(kp[i].pt, normalize_ray_dir=True)

        # Launch a ray from a point on the circle of diameter kp[i].size from
        # the point aligned with the feature's major axis.
        v = np.array(np.cos(kp[i].angle/180*np.pi),
                     np.sin(kp[i].angle/180*np.pi))
        pt2 = v*kp[i].size/2 +  kp[i].pt
        _, ray_dir2 = image.unproject(pt2, normalize_ray_dir=True)

        half_angle = np.arccos(float(np.dot(ray_dir2.T, zdir)))
        f = 1/np.tan(half_angle)

        # Find the component of ray_dir2 that is perpendicular to zdir.
        xdir = ray_dir2 - np.dot(ray_dir2.T, zdir)*zdir
        xdir /= np.linalg.norm(xdir)

        ydir = np.cross(zdir.T, xdir.T).T

        R = np.vstack([xdir.T, ydir.T, zdir.T])
        K = np.diag([f, f, 1])

        # K*R|T projection matrix.
        T = -np.dot(R, cam_pos)
        proj_matrix = np.dot(K, np.hstack([R, T]))

        if False:
            # Should be <0, 0, 1>.
            np.dot(P, np.vstack([cam_pos + 10*zdir, 1]))

            # Should be <0, 0, 1>.
            im_pt = np.dot(P, np.vstack([cam_pos + 10*ray_dir2, 1]))
            im_pt = im_pt[:2]/im_pt[2]

        image_features_.append(image_feature(kp[i].pt, image, des[i],
                                             proj_matrix, cam_pos,
                                             kp[i].size, half_angle*2))

    image_features[image_num] = image_features_


def likely_same_feature(feat1, feat2):
    xyz = cv2.triangulatePoints(feat1.proj_matrix,
                                feat2.proj_matrix, (0, 0), (0, 0))

    # Make sure the points are in front of the camera.
    xyz3 = xyz[:3]/xyz[3]
    v1 = xyz3 - feat1.view_pos
    v2 = xyz3 - feat2.view_pos

    dp = np.dot(v1.T, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    angle_baseline = np.arccos(float(dp))*180/np.pi

    if angle_baseline > 40:
        return False

    if np.dot(feat1.proj_matrix[2, :3], v1) < 0:
        return False

    if np.dot(feat2.proj_matrix[2, :3], v2) < 0:
        return False

    im_pti = np.dot(feat1.proj_matrix, xyz)
    im_pti = im_pti[:2]/im_pti[2]
    im_ptj = np.dot(feat2.proj_matrix, xyz)
    im_ptj = im_ptj[:2]/im_ptj[2]

    # The projection matrices map the sphere surrounding the real-world
    # feature into a unit circle. So, we will define good alignment as
    # mapping the triangulated point to within a circle with radius
    # 0.25.
    thresh = 0.5
    if (np.linalg.norm(im_pti) > thresh or
        np.linalg.norm(im_ptj) > thresh):
        return False

    # Check to see if the sizes make sense.
    d1 = np.linalg.norm(v1)
    d2 = np.linalg.norm(v2)

    # feat1 was viewed sd times closer.
    sd = d2/d1

    # feat1 manifested as sa times larger angular size.
    sa = feat1.angular_size/feat2.angular_size

    size_discrepancy_factor = sa/sd

    if size_discrepancy_factor < 1:
        # Make it so it always represents a value greater than
        # 1.
        size_discrepancy_factor = 1/size_discrepancy_factor

    if size_discrepancy_factor > 2:
        return False

    return True


# Define matchers.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                    key_size=12, multi_probe_level=1)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)


# Exhaustive matching.
image_nums = sorted(list(images.keys()))
for i in range(len(image_nums)):
    numi = image_nums[i]
    try:
        image_featuresi = image_features[numi]
    except KeyError:
        continue

    descri = np.array([feat.descriptor for feat in image_featuresi])
    for j in range(i + 1, len(image_nums)):
        print('Matching image %i to %i' % (i + 1, j + 1))
        numj = image_nums[j]

        try:
            image_featuresj = image_features[numj]
        except KeyError:
            continue

        descrj = np.array([feat.descriptor for feat in image_featuresj])

        if False:
            matches = [bf.match(descri, descrj)]
        else:
            # FLANN.
            matches = flann.knnMatch(descri, descrj, k=1)

        tic = time.time()
        for k, matches_ in enumerate(matches):
            for m in matches_:
                feat1 = image_featuresi[m.queryIdx]
                feat2 = image_featuresj[m.trainIdx]

                if likely_same_feature(feat1, feat2):
                    feat1.matches.append(feat2)
                    feat2.matches.append(feat1)

        print((time.time() - tic))

            #print(m.queryIdx, m.trainIdx)

        if False:
            # Visualize:
            imgi = images[numi].image.copy()
            imgj = images[numj].image.copy()

            for feat in image_features[numi]:
                for feat2 in feat.matches:
                    if feat2.image.image_id == numj:
                        cv2.circle(imgi,
                                   tuple(np.round(feat.im_pt).astype(np.int)),
                                   int(feat.size_pixels),  color=(255, 0, 255),
                                   thickness=3)

                        cv2.circle(imgj,
                                   tuple(np.round(feat2.im_pt).astype(np.int)),
                                   int(feat2.size_pixels),  color=(255, 0, 255),
                                   thickness=3)

            plt.close('all')
            figi = plt.figure()
            plt.imshow(imgi)
            figj = plt.figure()
            plt.imshow(imgj)









kp, des = orb.compute(img, kp)

    # Save images with keypoints superimposed. This allows selection of
    # pixels near keypoints to be geolocated.

    try:
        os.makedirs(georegister_data_dir)
    except OSError:
        pass

    for image_num in images:
        image = images[image_num]
        img_fname = '%s/%s' % (image_dir, image.name)
        img = cv2.imread(img_fname)

        for i, xy in enumerate(image.xys):
            if image.point3D_ids[i] == -1:
                continue

            xy = tuple(np.round(xy).astype(np.int))
            cv2.circle(img, xy, 5, color=(0, 0, 255), thickness=1)

        img_fname = '%s/images/%s.jpg' % (georegister_data_dir, image.id)
        img = cv2.imwrite(img_fname, img)


pts_3d = read_points3d_binary(points_3d_bin_fname)



ind = 341
img_fname = '%s/%s' % (image_dir, images[ind].name)
img_ref = cv2.imread(img_fname)
img_ref = cv2.fastNlMeansDenoisingColored(img_ref, None, 10, 10, 7, 21)
kp_ref, des_ref = orb.detectAndCompute(img_ref, None)

if False:
    pts, des0 = zip(*features[ind])
    pts = np.array(pts)
    des0 = np.array(des0)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.match(des, des_ref)

im_pts = []
im_pts_ref = []

for i, m in enumerate(matches):
    im_pts.append(kp[m.queryIdx].pt)
    im_pts_ref.append(kp_ref[m.trainIdx].pt)

im_pts = np.array(im_pts, dtype=np.float32)
im_pts_ref = np.array(im_pts_ref, dtype=np.float32)

if False:
    F, mask = cv2.findFundamentalMat(im_pts, im_pts_ref, cv2.FM_RANSAC)
else:
    H, mask = cv2.findHomography(im_pts, im_pts_ref, method=cv2.RANSAC,
                                 ransacReprojThreshold=1)

mask = mask.ravel() == 1
good_matches = [matches[_] for _ in range(len(matches)) if mask[_]]

# Draw first 10 matches.
plt.imshow(cv2.drawMatches(img, kp, img_ref, kp_ref, good_matches, None))

# We select only inlier points
im_pts = im_pts[mask.ravel() == 1]
im_pts_ref = im_pts_ref[mask.ravel() == 1]


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]

    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1 = img1.copy()

    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2 = img2.copy()

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines = cv2.computeCorrespondEpilines(im_pts.reshape(-1,1,2), 2, F)
lines = lines.reshape(-1,3)
img5,img6 = drawlines(img, img_ref, lines, im_pts, im_pts_ref)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()








    inds = [np.argmin(np.sum((image.xys - np.array(_.pt))**2, 1)) for _ in kp]
    ds = [np.sqrt(np.sum((image.xys[inds[i]] - np.array(kp[i].pt))**2))
          for i in range(len(inds))]


    image.xys

    img_fname = '%s/%s' % (image_dir, image.name)
    img = cv2.imread(img_fname)

    for i, xy in enumerate(image.xys):
        if image.point3D_ids[i] == -1:
            continue

        xy = tuple(np.round(xy).astype(np.int))
        cv2.circle(img, xy, 5, color=(0, 0, 255), thickness=1)

    features




if False:
    # Draw manual key points.
    save_dir = '/home/mattb/projects/calamityville/'
    for key in manual_matches:
        image_id1,image_id2 = key
        img1 = cv2.imread(image_fnames[image_id1 - 1])[:,:,::-1].copy()
        img2 = cv2.imread(image_fnames[image_id2 - 1])[:,:,::-1].copy()

        # These are the manually selected coordinates.
        kps = np.round(manual_matches[key]).astype(np.int)
        kp1s = kps[:,:2]
        kp2s = kps[:,2:]

        for i in range(len(kp1s)):
            cv2.circle(img1, (kp1s[i][0],kp1s[i][1]), 5, (255,0,255), 2)
            cv2.circle(img2, (kp2s[i][0],kp2s[i][1]), 5, (255,0,255), 2)

        fname1 = os.path.split(image_fnames[image_id1 - 1])[-1]
        fname2 = os.path.split(image_fnames[image_id2 - 1])[-1]
        cv2.imwrite(save_dir + fname1, img1[:,:,::-1])
        cv2.imwrite(save_dir + fname2, img2[:,:,::-1])


db = COLMAPDatabase.connect(database_path)
cursor = db.cursor()


if False:
    keep_pairs = set()
    # Remove matches with insufficient inliers.
    min_num_matches = 20
    cursor.execute("SELECT pair_id, data FROM two_view_geometries")
    for row in cursor:
        pair_id = row[0]
        if row[1] is not None:
            inlier_matches = np.fromstring(row[1],
                                           dtype=np.uint32).reshape(-1, 2)
            if len(inlier_matches) > min_num_matches:
                keep_pairs.add(pair_id)

    all_pairs = [pair_id
                 for pair_id, _ in db.execute("SELECT pair_id, data FROM matches")]

    for pair_id in all_pairs:
        if pair_id not in keep_pairs:
            print('Deleting pair:', pair_id)
            db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))


# Add missing keypoints.
for key in manual_matches:
    keypoints = dict((image_id, blob_to_array(data, np.float32, (-1, 2)))
                     for image_id, data in db.execute(
                     "SELECT image_id, data FROM keypoints"))

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
        if data is not None)

    image_id1,image_id2 = key
    keypoints1 = keypoints[image_id1]
    keypoints2 = keypoints[image_id2]

    # These are the manually selected coordinates.
    kp1 = manual_matches[key][:,:2]
    kp2 = manual_matches[key][:,2:]

    for kp in kp1:
        d = np.sqrt(np.sum((kp - keypoints1)**2, 1))
        if d.min() > 2:
            keypoints1 = np.vstack([keypoints1,kp])

    for kp in kp2:
        d = np.sqrt(np.sum((kp - keypoints2)**2, 1))
        if d.min() > 2:
            keypoints2 = np.vstack([keypoints2,kp])

    # Remove old set of keypoints
    db.execute("DELETE FROM keypoints WHERE image_id=?", (image_id1,))
    db.execute("DELETE FROM keypoints WHERE image_id=?", (image_id2,))

    db.add_keypoints(image_id1, keypoints1.copy())
    db.add_keypoints(image_id2, keypoints2.copy())


# Rebuild keypoint dictionary.
keypoints = dict((image_id, blob_to_array(data, np.float32, (-1, 2)))
                 for image_id, data in db.execute(
                 "SELECT image_id, data FROM keypoints"))

# Assing manual matches to keypoints.
for key in manual_matches:
    image_id1,image_id2 = key
from colmap_processing.static_camera_model import load_static_camera_from_file
    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
        if data is not None)

    try:
        matches = matches[(image_id1,image_id2)]
    except:
        matches = np.zeros((0,2), dtype=np.int)

    keypoints1 = keypoints[image_id1]
    keypoints2 = keypoints[image_id2]

    # These are the manually selected coordinates.
    kp1 = manual_matches[key][:,:2]
    kp2 = manual_matches[key][:,2:]

    if False:
        img1 = cv2.imread(image_fnames[image_id1 - 1])[:,:,::-1]
        img2 = cv2.imread(image_fnames[image_id2 - 1])[:,:,::-1]

        plt.figure()
        plt.imshow(img1)
        #plt.plot(keypoints1[:,0], keypoints1[:,1], 'ro')
        plt.plot(kp1[:,0], kp1[:,1], 'go')

        plt.figure()
        plt.imshow(img2)
        #plt.plot(keypoints2[:,0], keypoints2[:,1], 'ro')
        plt.plot(kp2[:,0], kp2[:,1], 'go')

        m = matches[(image_id1,image_id2)][:20]
        plt.figure()
        plt.imshow(img1)
        plt.plot(keypoints1[m[:,0],0], keypoints1[m[:,0],1], 'ro')
        plt.figure()
        plt.imshow(img2)
        plt.plot(keypoints2[m[:,0],0], keypoints2[m[:,0],1], 'ro')

    for i in range(len(kp1)):
        d = np.sqrt(np.sum((kp1[i] - keypoints1)**2, 1))
        ind1 = np.argmin(d)
        assert d[ind1] < 2

        d = np.sqrt(np.sum((kp2[i] - keypoints2)**2, 1))
        ind2 = np.argmin(d)
        assert d[ind2] < 2

        # Loop to artificially increase confidence.
        for _ in range(10):
            matches = np.vstack([matches,[ind1,ind2]])

    pair_id = image_ids_to_pair_id(image_id1, image_id2)
    db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
    db.add_matches(image_id1, image_id2, matches.copy())

    print('Adding manually registered matches to pair:', image_id1, image_id2)


# Commit the data to the file.
db.commit()

# Clean up.

db.close()