# =======================================
# Uncomment and run to tests superresolution model.

# import cv2
# import numpy as np
# from modules.sruperresolution import superresolution
# import os
# image_path = "/home/aru/WMG/DITMO/image_resized/a0636_rsz.png"
# save_dir = "check_folder"
# model_path = 'utils/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
# scale = 4
# window_size = 8

# # read image
# (imgname, imgext) = os.path.splitext(os.path.basename(image_path))
# img_lq = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

# img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
# output = superresolution(img_lq)
# cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)
# ======================================

# ===================================
# Uncomment and run to tests santos model.


import numpy as np
from modules.santos import get_saturated_regions, writeEXR, writeLDR, santos
from utils.ditmo_utils import resize_and_pad_image

image_path = "/home/aru/WMG/DITMO/image_resized/a0636_rsz.png"
image = resize_and_pad_image(image_path)
# load image
# image = load_image(image_path)
image = np.asarray(image.convert('RGB')).astype(np.float32)/255.0
# get saturation mask
conv_mask = 1 - get_saturated_regions(image)

H, mask = santos(image, conv_mask)
writeEXR(H, "check_folder/img.exr")
writeLDR(mask, "check_folder/img.png")
# =======================================