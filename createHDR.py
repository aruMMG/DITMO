import os
import cv2
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

fname = "a0879"
addr = f"/home/goswam_a@WMGDS.WMG.WARWICK.AC.UK/ditmo/stable-diffusion-main/data/fivek/hal_rgb/{fname}_v2/"

img_fn = [f"{fname}_rsz.png", f"{fname}_0.png", f"{fname}_1.png", f"{fname}_2.png"]
img_list = [cv2.imread(f"{addr}{fn}") for fn in img_fn]
exposure_times = np.array([30, 15.0, 7.5, 3.75], dtype=np.float32)
merge_debevec = cv2.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
cv2.imwrite(f"{addr}{fname}_mergedhdr.exr", hdr_debevec)