import numpy as np
from matplotlib import pyplot as plt
import cv2


def imshow(image):
    plt.imshow(image)
    plt.show()


addr = "/home/agoswami/PycharmProjects/diffusioninpainting/stable-diffusion/data/inpainting_examples"

orig_im = cv2.cvtColor(cv2.imread(f'{addr}/valley_sq.png'), cv2.COLOR_BGR2RGB)
hal_im = cv2.cvtColor(cv2.imread(f'{addr}/valley_hal_210_cloudysky.png'), cv2.COLOR_BGR2RGB)

mask = cv2.imread(f'{addr}/valley_mask_210.png')[:,:,0]
inv_mask = cv2.bitwise_not(mask)


hal_mask = cv2.bitwise_and(hal_im,hal_im,mask=mask)
rest_mask = cv2.bitwise_and(orig_im,orig_im,mask=inv_mask)

merged = hal_mask + rest_mask


imshow(hal_mask)
imshow(rest_mask)
imshow(merged)