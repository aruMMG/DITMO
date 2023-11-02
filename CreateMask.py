import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob
import os


def imshow(img):
    plt.imshow(img)
    plt.show()


addr = "./data/fivek/image fs/"
save_im_addr = "./data/fivek/image resized/"
save_rgb_mask_addr = "./data/fivek/rgb_mask/"
save_bin_mask_addr = "./data/fivek/bin_mask/"


dim = (512, 512)


for file in glob(addr + '*jpg'):
    fname = os.path.basename(file).split('.')[0]
    print(fname)
    im_bgr = cv2.imread(file)
    inp_bgr = cv2.resize(im_bgr, dim, interpolation=cv2.INTER_AREA)
    inp_rgb = cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB)

    inp_gray = cv2.cvtColor(inp_rgb, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(inp_gray, 225, 255, cv2. THRESH_BINARY_INV)
    rgb_mask = cv2.bitwise_and(inp_bgr, inp_bgr, mask=threshold)

    threshold = cv2.bitwise_not(threshold)

    # # imshow(inp_bgr)
    # cv2.imwrite(f"{save_im_addr}{fname}_rsz.png", inp_bgr)
    # imshow(threshold)
    cv2.imwrite(f"{save_bin_mask_addr}{fname}_binmask.png", threshold)
    # # imshow(rgb_mask)
    # cv2.imwrite(f"{save_rgb_mask_addr}{fname}_rgbmask.png", rgb_mask)
