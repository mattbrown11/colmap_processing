import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

fnames = glob.glob('/mnt/ten_tb_data/kamera/Calibration imagery/fl08/colmap/images0/fl09_right_uv/*.jpg')

stretch_percentiles = [0.1, 99.9]
monochrome = True
clip_limit = 10
median_blur_diam = 3

for fname in fnames:
    img = cv2.imread(fname)
    if img.ndim == 3 and monochrome:
        img = img[:, :, 0]

    img = img.astype(np.float)
    img -= np.percentile(img.ravel(), stretch_percentiles[0])
    img[img < 0] = 0
    img /= np.percentile(img.ravel(), stretch_percentiles[1])/255
    img[img > 255] = 255
    img = np.round(img).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    img = clahe.apply(img)

    img = cv2.medianBlur(img, median_blur_diam)
    cv2.imwrite(fname, img)
