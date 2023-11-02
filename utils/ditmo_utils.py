from matplotlib import pyplot as plt
import numpy as np
import cv2


def imshow(image, title="image"):
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.show()

def prune_mask(ori_mask, l):
    mod_mask = np.zeros_like(ori_mask)
    mod_mask = np.where(ori_mask == l, 255.0, mod_mask)
    return mod_mask


def get_saturated_mask(img_rgb):
    inp_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    _, saturated_mask = cv2.threshold(inp_gray, 225, 255, cv2.THRESH_BINARY_INV)
    return saturated_mask


def saturated_perc(hal_im, modified_mask):
    hal_mask = cv2.bitwise_and(hal_im, hal_im, mask=modified_mask)
    imshow(hal_mask, "halucinated")
    # perc = 0
    # return perc


def get_semantic_inpainting_mask_list(threshold_segment_mask):
    list_of_seg = np.unique(threshold_segment_mask)
    mod_list_of_seg = []
    print(list_of_seg)
    total_mask = cv2.countNonZero(threshold_segment_mask)
    for l in list_of_seg[1:]:
        perc = np.count_nonzero(threshold_segment_mask == l) * 100.0 / total_mask
        print(f"{l} ---> " + str(perc))
        if perc > 10:
            mod_list_of_seg.append(l)
    print(mod_list_of_seg)
    return mod_list_of_seg


def get_lower_bracket(im, factor):
    im = im.astype('float32')
    im = (im ** 2.2) / 2**factor
    im = im ** (1/2.2)
    return im