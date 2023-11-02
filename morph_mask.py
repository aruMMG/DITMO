import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import cv2
import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from skimage import morphology
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa


def imshow(image, title="image"):
    plt.figure()
    plt.title(title)
    plt.imshow(image)


out_dir = "/home/agoswami/PycharmProjects/diffusioninpainting/stable-diffusion/data/fivek/hal_rgb/"
mask_dir = "/home/agoswami/PycharmProjects/diffusioninpainting/stable-diffusion/data/fivek/bin_mask/"
mask_im = cv2.imread(f"{mask_dir}a0418_binmask.png")

imshow(mask_im)
# kernel = np.ones((15, 15), np.uint8)
# img_erosion = cv2.erode(mask_im, kernel, iterations=1)
# img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
# imshow(img_erosion, "cv2 eroded")
# imshow(img_dilation, " cv2 dilated")
footprint5 = disk(2)
footprint10 = disk(15)
eroded = erosion(img_as_ubyte(mask_im[:,:,0]),footprint10)*255.0
dilated = dilation(eroded, footprint10)
print(eroded.shape)
print(type(eroded))

cv2.imwrite(f'{out_dir}eroded_mask.png', eroded.astype('uint8'))
imshow(eroded, "scikit eroded")
# imshow(dilated, "scikit dilated")
plt.show()