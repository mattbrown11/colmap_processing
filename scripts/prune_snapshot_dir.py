import glob
import shutil
import time

snap_shot_dir = '/media/root/LaCie/AP_Hill_2020_Oct/DP3/colmap/snapshots/*'
num_to_keep = 10

while True:
    dirs = glob.glob(snap_shot_dir)
    dirs = sorted(dirs)
    for d in dirs[:-num_to_keep]:
        shutil.rmtree(d)
        
    time.sleep(10)