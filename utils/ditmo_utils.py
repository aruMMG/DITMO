from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
from PIL import Image
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
from modules.sruperresolution import superresolution
from pymatting import *
import imageio
import sys
import re

sat_thresh = 245
rad1 = disk(1)
rad2 = disk(2)
rad5 = disk(5)
rad10 = disk(10)
rad20 = disk(20)
rad50 = disk(50)
temp_dir = "temp/"


def imshow_sep(image, title="image"):
    plt.figure()
    plt.title(title)
    plt.imshow(image)


def imshow(image, title="image"):
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.show()


def pil_to_opencv(image_pil):
    # Convert PIL Image to NumPy array (OpenCV format)
    image_cv = np.array(image_pil)

    # Convert RGB to BGR (OpenCV uses BGR color order)
    if image_cv.shape[-1] == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    return image_cv


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
    total_mask = cv2.countNonZero(threshold_segment_mask)
    for l in list_of_seg[1:]:
        perc = np.count_nonzero(threshold_segment_mask == l) * 100.0 / total_mask
        # print(f"{l} ---> " + str(perc))
        if perc > 10:
            mod_list_of_seg.append(l)
    # print(mod_list_of_seg)
    return mod_list_of_seg


def resize_and_pad_centre_align(image, target_width=1920, target_height=1080):
    # Get the original image dimensions
    image = image.astype('float32')
    original_height, original_width = image.shape[:2]

    # Calculate the scaling factor to fit within the target dimensions
    scale_factor = min(target_width / original_width, target_height / original_height)

    # Calculate the new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image while preserving the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a blank canvas of the target dimensions
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.float32)

    # Calculate the position to center the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Paste the resized image onto the padded canvas
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image


def resize_and_pad_image(img, target_size=512):
    """
    Take a input image and return resized image to target size by keeping aspect ratio same. To achive the target size it pad on the smaller side.
    Input:
        img: the input image.
        target_size

    output:
        return resized and padded image in PIL format.
    """
    pad = [0, 0]
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
        pad[0] = new_height
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
        pad[1] = new_width

    resized_img = img.resize((new_width, new_height))

    pad_width = target_size - resized_img.width
    pad_height = target_size - resized_img.height

    padded_img = ImageOps.expand(resized_img, border=(0, 0, pad_width, pad_height), fill=0)

    return padded_img, pad


def remove_pad(im, padding_dim):
    target_dim = 512
    if padding_dim[0] == 0 and padding_dim[1] != 0:
        cropped_im = im[0:(target_dim - padding_dim[0]), 0:(padding_dim[1])]
    elif padding_dim[1] == 0 and padding_dim[0] != 0:
        cropped_im = im[0:padding_dim[0], 0:(target_dim - padding_dim[1])]
    else:
        cropped_im = im
    return cropped_im


# ------------------------------------------------------------------------ Alternate dtypes
# def createHDR(im_ori, im_hal, t_ori=1, t_hal=.5):
#     im_ori = im_ori.astype('float32')/255.0
#     im_hal = im_hal.astype('float32')
#     img_list = [im_ori, im_hal]
#     print("Merging Debevec.........")
#     imshow(im_ori, "inside CreateHDR - Original")
#     imshow(im_hal, "inside CreateHDR - Hal")
#
#     # for img in img_list:
#     #     print(img.shape)
#     #     print(img.dtype)
#     #     print(img.depth())
#     exposure_times = np.array([t_ori, t_hal], dtype=np.float32)
#     merge_debevec = cv2.createMergeDebevec()
#     hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
#     return hdr_debevec


def createHDR_multibrack(list_im, list_times):
    # im_ori = im_ori.astype(np.uint8)
    # im_hal = im_hal.astype(np.uint8)
    # im_ori = im_ori.astype(np.float32)
    # im_hal = im_hal.astype(np.float32)
    img_list = [i.astype(np.uint8) for i in list_im]
    print("Merging Debevec Multibrack.........")
    # imshow(im_ori, "inside CreateHDR - Original")
    # imshow(im_hal, "inside CreateHDR - Hal")

    # for img in img_list:
    #     print(img.shape)
    #     print(img.dtype)
    #     print(img.depth())
    exposure_times = np.array(list_times, dtype=np.float32)
    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
    return hdr_debevec


def createHDR(im_ori, im_hal, t_ori=1, t_hal=0.5):
    im_ori = im_ori.astype(np.uint8)
    im_hal = im_hal.astype(np.uint8)
    # im_ori = im_ori.astype(np.float32)
    # im_hal = im_hal.astype(np.float32)
    img_list = [im_ori, im_hal]
    print("Merging Debevec.........")
    # imshow(im_ori, "inside CreateHDR - Original")
    # imshow(im_hal, "inside CreateHDR - Hal")

    # for img in img_list:
    #     print(img.shape)
    #     print(img.dtype)
    #     print(img.depth())
    exposure_times = np.array([t_ori, t_hal], dtype=np.float32)
    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
    return hdr_debevec


def smart_extend(im):
    im_lowres = cv2.resize(im, (0, 0), fx=.2, fy=.2, interpolation=cv2.INTER_AREA)
    rgb_dilated = im_lowres.copy()
    rgb_dilated[:, :, 0] = dilation(rgb_dilated[:, :, 0], rad5)
    rgb_dilated[:, :, 1] = dilation(rgb_dilated[:, :, 1], rad5)
    rgb_dilated[:, :, 2] = dilation(rgb_dilated[:, :, 2], rad5)
    rgb_dilated_hires = cv2.resize(rgb_dilated, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    rgb_dilated_hires = np.where(im == 0, rgb_dilated_hires, im)
    return rgb_dilated_hires


def get_halucinatedrgb(mask, hal_im):
    mask = mask2channels(mask)
    just_hal = np.where(mask == 255, hal_im, mask)
    # just_hal = cv2.cvtColor(just_hal, cv2.COLOR_BGR2RGB)
    return just_hal


def dilate_mask(mmask):
    # du.imshow(mmask, 'unfiltered mask')
    # eroded = erosion(img_as_ubyte(mmask), footprint10)
    dilated = dilation(mmask.astype('uint8'), rad5)
    dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
    # du.imshow(dilated, "dilated mask")
    return dilated


def merge_blur(hal_im, im, mmask):
    # footprint10 = disk(15)
    # eroded = erosion(img_as_ubyte(mmask[:, :, 0]), footprint10)
    # dilated = dilation(eroded, footprint10)
    # dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
    ksize = (5, 5)
    # # blur_mask = cv2.bilateralFilter(mask, 10, 150,150)
    hal_blur_mask = cv2.blur(mmask, ksize) / 255.0
    rest_blur_mask = 1 - hal_blur_mask

    merge_im = (rest_blur_mask * im + hal_blur_mask * hal_im).astype('uint8')
    return merge_im


def create_saturated_mask(hal_img, stage0_mask, saturation_threshold=sat_thresh):
    """this will find the saturated pixels left in the haluginated image for a specific class.
    Find the saturated pixels in the mask area and if it crosses the 5% threshold then return a mask image or return None.
    """
    saturated_mask = (hal_img > saturation_threshold).astype(np.uint8)

    # Apply the mask from the input mask image
    saturated_mask = saturated_mask * 255
    opencv_mask_image = mask2channels(stage0_mask)

    saturated_mask = saturated_mask * (opencv_mask_image == 255)

    return saturated_mask[:, :, 0]


def create_saturated_mask_stage0(input_image, saturation_threshold=225):
    """this will find the saturated pixels and return a binary image with saturated pixel as white (255) and not saturated as black (0)"""
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # du.imshow(gray_image, "gray im")
    saturated_pixels = (gray_image > saturation_threshold).astype(np.uint8) * 255

    return saturated_pixels


def mask2channels(mask_image):
    opencv_mask_image = np.zeros((mask_image.shape[0], mask_image.shape[1], 3), dtype=np.uint8)

    # Copy the single channel into all three color channels
    opencv_mask_image[:, :, 0] = mask_image
    opencv_mask_image[:, :, 1] = mask_image
    opencv_mask_image[:, :, 2] = mask_image
    return opencv_mask_image


def apply_hallucination_masks(hal_images, original_img, mask_images):
    """
    Input:
        hal_images: list of hallucinated images in array format (like reading with opencv)
        original_img: the original image to add the hallucination.
        mask_images: list of coresponding mask images in array format (like reading with opencv)

    output:
        result_image: a image with hallucinated features.

    The original images for stage zero will be the LDR image, For other stages it should be exposure corrected hallucinated images.
    """
    if len(hal_images) != len(mask_images):
        raise ValueError("hal_images and mask_images lists must have the same length")

    result_image = np.copy(original_img)

    for hal_image, mask_image in zip(hal_images, mask_images):
        if isinstance(hal_image, Image.Image):
            opencv_hal_image = pil_to_opencv(hal_image)
        if isinstance(mask_image, Image.Image):
            opencv_mask_image = pil_to_opencv(mask_image)
        opencv_mask_image = mask2channels(mask_image)
        if opencv_hal_image.shape != original_img.shape or opencv_mask_image.shape != original_img.shape:
            raise ValueError("All images in hal_images, original_img, and mask_images should have the same dimensions")

        result_image = np.where(opencv_mask_image == 255, opencv_hal_image, result_image)

    return result_image


def apply_hallucination_masks_blf(hal_images, original_img, mask_images):
    """
    Input:
        hal_images: list of hallucinated images in array format (like reading with opencv)
        original_img: the original image to add the hallucination.
        mask_images: list of coresponding mask images in array format (like reading with opencv)

    output:
        result_image: a image with hallucinated features.

    The original images for stage zero will be the LDR image, For other stages it should be exposure corrected hallucinated images.
    """
    if len(hal_images) != len(mask_images):
        raise ValueError("hal_images and mask_images lists must have the same length")

    result_image = np.copy(original_img)

    for hal_image, mask_image in zip(hal_images, mask_images):
        if isinstance(hal_image, Image.Image):
            opencv_hal_image = pil_to_opencv(hal_image)
            opencv_hal_image = cv2.cvtColor(opencv_hal_image, cv2.COLOR_BGR2RGB)
        else:
            opencv_hal_image = hal_image
            opencv_hal_image = cv2.cvtColor(opencv_hal_image, cv2.COLOR_BGR2RGB)
        if isinstance(mask_image, Image.Image):
            opencv_mask_image = pil_to_opencv(mask_image)
        else:
            opencv_mask_image = mask_images
        opencv_mask_image = mask2channels(mask_image)
        if opencv_hal_image.shape != original_img.shape or opencv_mask_image.shape != original_img.shape:
            raise ValueError("All images in hal_images, original_img, and mask_images should have the same dimensions")

        # result_image = np.where(opencv_mask_image == 255, opencv_hal_image, result_image)
        # result_image = colorBlending(opencv_hal_image, result_image, mask_image)
        print("inside merging....")
        result_image = merge_blur(opencv_hal_image, result_image, opencv_mask_image)
    return result_image


def unsure_trimap(tmap):
    unsure = np.zeros_like(tmap)
    unsure = np.where(tmap <= 128, 255, unsure)
    return unsure


def get_centroid(mask):
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print(f"center X : {cX}")
    print(f"center Y : {cY}")

    # Draw a circle based centered at centroid coordinates
    # cv2.circle(mask, (round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])), 5, (0, 255, 0), -1)
    # cv2.imshow("outline contour & centroid", mask)
    return cX, cY


def get_rectbox_centre(mask):
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    tv, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contour = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cent_x, cent_y = 0, 0
    for c in contour:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cent_x = x + w // 2
        cent_y = y + h // 2
    return cent_x, cent_y


def extend_by_overlap(im):
    im_low = im.copy()
    w, h = im.shape[1], im.shape[0]
    im_high = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    W, H = im_high.shape[1], im_high.shape[0]

    # ------------ Super res (input im has to be 512x512)
    # im_high = upscale_superresx4(im_high)

    # ----------------- Naive upsample
    im_high = cv2.resize(im_high, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    # ------------ Align by centroid of mask
    # cx, cy = get_centroid(im_low)
    # Cx, Cy = get_centroid(im_high)
    # layer = im_high[(Cy - cy):(Cy - cy + h), (Cx - cx):(Cx - cx + w)]

    # ---------- align by centre of image
    # layer = im_high[int((H - h) / 2):int((H - h) / 2 + h), int((W - w) / 2):int((W - w) / 2 + w)]

    # ------------- align by top of image
    # layer = im_high[0:h, int((W - w) / 2):int((W - w) / 2 + w)]

    # --------------------------------- align by centre of bounding box
    cx, cy = get_rectbox_centre(im_low)
    Cx, Cy = get_rectbox_centre(im_high)
    layer = im_high[(Cy - cy):(Cy - cy + h), (Cx - cx):(Cx - cx + w)]

    # --------------------------------SHOW
    layer = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
    # overlap = np.where(im_low > 0, im_low, layer)
    return layer


def upscale_superresx4(im):
    model_path = 'utils/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
    scale = 2
    window_size = 8
    # read image
    # im = im.astype(np.float32)
    # im = cv2.resize(im, (512, 512))
    lowres = np.transpose(im if im.shape[2] == 1 else im[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    hires = superresolution(lowres)
    return hires


def smart_merge(alpha_softmask, hal_tomerge, original_image):
    # hal_extended = extend_by_overlap(hal_tomerge)
    soft_mask3ch = cv2.cvtColor(alpha_softmask, cv2.COLOR_GRAY2RGB)
    merged = original_image * (1 - soft_mask3ch) + hal_tomerge / 255.0 * soft_mask3ch
    return merged


def apply_smart_merge(hal_tomerge_images, original_img, soft_mask_images):
    if len(hal_tomerge_images) != len(soft_mask_images):
        raise ValueError("hal_images and mask_images lists must have the same length")

    result_image = np.copy(original_img)

    for hal_image, soft_mask in zip(hal_tomerge_images, soft_mask_images):
        hal_image = hal_image
        soft_mask = soft_mask
        hal_extended = extend_by_overlap(hal_image)

        print("inside merging....")
        imshow(hal_image, "hal_image")
        imshow(soft_mask, "soft_mask")
        imshow(hal_extended, "hal_extended")
        soft_mask3ch = cv2.cvtColor(soft_mask, cv2.COLOR_GRAY2RGB)

        # merged_hal = (hal_extended/255.0) * soft_mask3ch
        # merged_non_hal = original_img * (1-soft_mask3ch)
        result_image = result_image * (1 - soft_mask3ch) + (hal_extended / 255.0) * soft_mask3ch
        imshow(result_image, "Result inside Smart_merge")
    return result_image.astype('float32') * 255.0


def merge(hal_tomerge_images, original_img, soft_mask_images):
    if len(hal_tomerge_images) != len(soft_mask_images):
        raise ValueError("hal_images and mask_images lists must have the same length")

    result_image = np.copy(original_img)
    # result_image = result_image.astype('float32') / 255.0
    result_image = result_image.astype('float32')


    for hal_image, soft_mask in zip(hal_tomerge_images, soft_mask_images):
        # imshow(hal_image)
        hal_image = hal_image.astype('float32') / 255.0
        soft_mask = soft_mask

        # print("Merging halucinations....")
        # imshow(hal_image, "hal_image")
        # imshow(soft_mask, "soft_mask")
        # imshow(result_image, "Original")

        soft_mask3ch = cv2.cvtColor(soft_mask, cv2.COLOR_GRAY2RGB)

        # merged_hal = (hal_extended/255.0) * soft_mask3ch
        # merged_non_hal = original_img * (1-soft_mask3ch)
        result_image = result_image * (1 - soft_mask3ch) + hal_image * soft_mask3ch
        # imshow(result_image, "Result inside Merge")
    return result_image.astype('float32')


def createmattes_py(ref_im, binary_mask, border_t=10, scale=0.5):
    ref_im_file_padded_path = f"{temp_dir}refim_temp.png"
    trimap_file_padded_path = f"{temp_dir}trimap_temp.png"
    # scale = 1
    bin_im_shape = binary_mask.shape
    bm_downsample = cv2.resize(binary_mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    trimap = morph_expand(bm_downsample, border_t)
    trimap_upsampled = cv2.resize(trimap, (bin_im_shape[1], bin_im_shape[0]), interpolation=cv2.INTER_NEAREST)
    imageio.imsave(ref_im_file_padded_path, ref_im)
    imageio.imsave(trimap_file_padded_path, trimap_upsampled)

    ref_im = load_image(ref_im_file_padded_path, "RGB", 1, "box")
    trimap_input = load_image(trimap_file_padded_path, "GRAY", 1, "nearest")
    alpha = estimate_alpha_cf(ref_im, trimap_input)
    return alpha


def morph_expand(mask, border=10):
    """
    Morphologically expand the segment to create trimap.
    :param mask: takes the input segment to be morphologically dilated
    :param border: dilation parameter. Thicker border for segments with high frequency on the border
    :return: Dilated trimap
    """

    shape = disk
    border = border * np.sqrt(2)

    trimap = np.zeros(mask.shape, dtype=np.uint8)
    NSURE = 128

    # Max, Min = 255, 0
    Max, Min = np.max(mask), np.min(mask)

    selem = shape(border)
    # maskM = dilation(mask == Max, selem=selem)
    # maskm = dilation(mask == Min, selem=selem)
    # maskU = dilation(mask == NSURE, selem=selem)

    maskM = dilation(mask == Max, disk(border))
    maskm = dilation(mask == Min, disk(0))
    maskU = dilation(mask == NSURE, disk(border))

    trimap[mask == Max] = 255
    trimap[mask == Min] = 0

    trimap[maskM & maskm] = NSURE
    trimap[maskU] = NSURE

    return trimap


def reduce_exposure(im, stops=1):
    print(f'reducing by {stops} stops')
    im = im.astype('float32') / 255.0
    im = (im ** 2.2) / (2 ** stops)
    im = im ** (1 / 2.2)
    return (im * 255.0).astype('uint8')


def increase_exposure(im, stops=1):
    # print(f'increasing by {stops} stops')
    im = im.astype('float32') / 255.0
    im = (im ** 2.2)
    # imshow(im, "im linear inside INC EXP")
    im = im * (2 ** stops)
    # imshow(im, "im+1stop linear inside INC EXP")
    im = im ** (1 / 2.2)
    # imshow(im, "im+1stop with gamma inside INC EXP")
    # im = np.clip(im, 0, 255)
    im = np.clip(im, 0, 1)
    return im


def modify_exposure(im, stops=1):
    # print(f'modifying by {stops} stops')
    im = im.astype('float32')
    im = (im ** 2.2) * (2 ** stops)
    im = im ** (1 / 2.2)
    return im


def modify_exposure_hdr(im, stops=1):
    # print(f'modifying by {stops} stops')
    im = im.astype('float32')
    im = im  * (2 ** stops)
    im = im ** (1 / 2.2)
    return im


def predict_bracket(original_image, binary_mask):
    # Load the images
    binary_mask = binary_mask // 255
    # imshow(original_image, "inside predict bracket - original_image")
    # imshow(binary_mask, "inside predict bracket - bin mask")
    lum_factor = np.array([.299, .587, .114])
    for i in range(10):

        higher_exposed_image = increase_exposure(original_image, i)
        # lum = np.matmul(higher_exposed_image, lum_factor)
        # hsv_image = cv2.cvtColor(higher_exposed_image, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(higher_exposed_image, cv2.COLOR_BGR2GRAY)
        # imshow(hsv_image, "hsv")
        # imshow(gray_image, "gray")

        saturated_mask = (gray_image > .97).astype(np.uint8)
        masked_saturated = cv2.bitwise_and(saturated_mask, binary_mask)
        total_pixels = np.sum(binary_mask)
        oversaturated_pixels = np.sum(masked_saturated)
        percentage_oversaturation = (oversaturated_pixels / total_pixels) * 100
        # print(f"percentage of sat at {i} stop is {percentage_oversaturation}%")
        # imshow(higher_exposed_image, f"Higher ex {i + 1}")
        if percentage_oversaturation > 97:
            # cv2.imwrite("check_folder/higher_exposed_image98.png", higher_exposed_image.astype(np.uint8))
            return i
    return False


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)


def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.search(r'(\d+)\s(\d+)', file.readline().decode('ascii'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape), scale


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape), scale


