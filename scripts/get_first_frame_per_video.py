from __future__ import division, print_function
import numpy as np
import os
import cv2
import subprocess
import matplotlib.pyplot as plt
import glob
import natsort

base_dir = '/media/matt.brown/UTB_Content/Raw_Content/20191216-20_URSA_UTB/20191220_Organized_Data'
out_dir = '/media/matt.brown/27f1e437-6e92-4f06-86c4-178c2cf8fe17/2019-12-18_darpa_testing_mutc/sfm_3d_reconstruction_images/darpa_data/all_videos'


list_of_files = []
for (dirpath, dirnames, filenames) in os.walk(base_dir):
    for filename in filenames:
        if (filename.endswith('.mp4') or filename.endswith('.MP4')
             or filename.endswith('.avi')):
            list_of_files.append(os.sep.join([dirpath, filename]))

for fname in list_of_files:
    cap = cv2.VideoCapture(fname)
    ret, frame = cap.read()
    if frame is not None:
        print(frame.shape)
    
    base = os.path.splitext(os.path.split(fname)[1])[0]
    fname = '%s/%s.png' % (out_dir, base)
    cv2.imwrite(fname, frame)
