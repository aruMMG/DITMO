from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps


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

def resize_and_pad_image(img, target_size=512):
    """
    Take a input image and return resized image to target size by keeping aspect ratio same. To achive the target size it pad on the smaller side.
    Input:
        img: the input image.
        target_size

    output:
        return resized and padded image in PIL format.
    """
    if not isinstance(img, Image.Image):
        try:
            if isinstance(img, str):
                img = Image.open(img)
            else:
                img = Image.fromarray(img)
        except Exception as e:
            print(f"Error in file format. The image instance is {type(img).__name__}. Exception: {str(e)}")
    aspect_ratio = img.width / img.height

    if img.width > img.height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    resized_img = img.resize((new_width, new_height))

    pad_width = target_size - resized_img.width
    pad_height = target_size - resized_img.height

    padded_img = ImageOps.expand(resized_img, border=(0, 0, pad_width, pad_height), fill=0)

    return padded_img

def get_higher_bracket(im, factor):
    im = im.astype('float32')
    im = (im ** 2.2) * factor
    im = im ** (1/2.2)
    im = np.clip(im, 0, 255)
    return im

def predict_bracket(original_image, binary_mask):
    # Load the images
    binary_mask = binary_mask // 255
    for i in range(5):

        higher_exposed_image = get_higher_bracket(original_image, 2**(i+1))
        hsv_image = cv2.cvtColor(higher_exposed_image, cv2.COLOR_BGR2HSV)

        saturated_mask = (hsv_image[:, :, 2] > 245).astype(np.uint8)

        masked_saturated = cv2.bitwise_and(saturated_mask, binary_mask)

        total_pixels = np.sum(binary_mask)
        oversaturated_pixels = np.sum(masked_saturated)
        percentage_oversaturation = (oversaturated_pixels / total_pixels) * 100

        if percentage_oversaturation > 98:
            # cv2.imwrite("check_folder/higher_exposed_image98.png", higher_exposed_image.astype(np.uint8))
            return i+1
    return False