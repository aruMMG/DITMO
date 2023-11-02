import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import cv2
import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import utils.ditmo_utils as du
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
from skimage.util import img_as_ubyte
from laplacianMarge import colorBlending


sat_thresh = 235
rad2 = disk(2)
rad5 = disk(5)
rad10 = disk(10)


def pil_to_opencv(image_pil):
    # Convert PIL Image to NumPy array (OpenCV format)
    image_cv = np.array(image_pil)

    # Convert RGB to BGR (OpenCV uses BGR color order)
    if image_cv.shape[-1] == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    return image_cv


def reduce_exposure(im, factor=1):
    im = im.astype('float32')/255.0
    im = (im ** 2.2) / (2 ** factor)
    im = im ** (1 / 2.2)
    return (im*255.0).astype('uint8')


# def reduce_exposure(image, exposure_factor=0.5):
#     """
#     Input:
#         image: image to reduce exposure in array format (like reading with opencv)
#         exposure_factor: The factor by which the exposure required to changed

#     output:
#         adjusted_image: The changed exposure image

#     The original images for stage zero will be the LDR image, For other stages it should be exposure corrected hallucinated images.
#     """

#     # Reduce all channel
#     reduced_image = image * exposure_factor
#     reduced_image = np.clip(reduced_image, 0, 255)
#     reduced_image = reduced_image.astype(np.uint8)
#     adjusted_image = reduced_image

#     # # Reduce Y channel only
#     # xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
#     # X, Y, Z = cv2.split(xyz_image)
#     # xyz_channels = [X, Y, Z]
#     # xyz_channels = [channel.astype(np.float32) for channel in xyz_channels]

#     # reduced_Y = xyz_channels[1] * exposure_factor
#     # reduced_Y = np.clip(reduced_Y, 0, 255)

#     # adjusted_xyz_image = np.stack(xyz_channels, axis=-1)
#     # adjusted_xyz_image[:, :, 1] = reduced_Y  # Replace Y channel

#     # adjusted_image = cv2.cvtColor(adjusted_xyz_image, cv2.COLOR_XYZ2BGR)

#     return adjusted_image

def dilate_mask(mmask):
    # du.imshow(mmask, 'unfiltered mask')
    # eroded = erosion(img_as_ubyte(mmask), footprint10)
    dilated = dilation(mmask.astype('uint8'), rad5)
    dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
    # du.imshow(dilated, "dilated mask")
    return dilated


def merge(hal_im, im, mmask):
    # footprint10 = disk(15)
    # eroded = erosion(img_as_ubyte(mmask[:, :, 0]), footprint10)
    # dilated = dilation(eroded, footprint10)
    # dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
    ksize = (15, 15)
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
        if isinstance(mask_image, Image.Image):
            opencv_mask_image = pil_to_opencv(mask_image)
        opencv_mask_image = mask2channels(mask_image)
        if opencv_hal_image.shape != original_img.shape or opencv_mask_image.shape != original_img.shape:
            raise ValueError("All images in hal_images, original_img, and mask_images should have the same dimensions")

        # result_image = np.where(opencv_mask_image == 255, opencv_hal_image, result_image)
        # result_image = colorBlending(opencv_hal_image, result_image, mask_image)
        result_image = merge(opencv_hal_image, result_image, opencv_mask_image)
    return result_image


def main(pipe, promptdict, lookuptable, fname, img_dir, seg_img_dir, out_dir):
    img = cv2.imread(f'{img_dir}{fname}_rsz.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    saturated_mask = create_saturated_mask_stage0(
        img)  # this will find the saturated pixels and return a binary image with saturated pixel as white (255) and not saturated as black (0)
    du.imshow(saturated_mask, " thres mask")
    seg_im = cv2.imread(f'{seg_img_dir}{fname}_rsz.png')[:, :, 0]  # Read the segmeted images. It has 150 classes where pixel value represent the class
    im_label = lookuptable[seg_im + 1]
    du.imshow(im_label*25, "seg mask")
    single_thres_mask_with_sep_class = cv2.bitwise_and(im_label, im_label, mask=saturated_mask)
    # du.imshow(single_thres_mask_with_sep_class, "seg mask")
    saturated_class_indexes = du.get_semantic_inpainting_mask_list(
        single_thres_mask_with_sep_class)  # Find the classes saturated

    # Iteration 1

    list_of_hal_imgs0 = []
    separate_class_masks0 = []
    input_img_to_pipe = img

    for saturated_class in saturated_class_indexes:
        one_class_mask = du.prune_mask(single_thres_mask_with_sep_class,
                                       saturated_class)  # Full mask of one class and remove other saturated mask areas.
        eroded = erosion(img_as_ubyte(one_class_mask.astype('uint8')), rad2)
        one_class_mask = dilation(eroded, rad5)
        mask_img_to_pipe = one_class_mask
        print(saturated_class, promptdict[saturated_class])
        # du.imshow(Image.fromarray(mask_img_to_pipe), 'mask entering pipe')
        # du.imshow(Image.fromarray(input_img_to_pipe), 'image entering pipe')

        hal_image = pipe(prompt=promptdict[saturated_class], image=Image.fromarray(input_img_to_pipe),
                         mask_image=Image.fromarray(mask_img_to_pipe)).images[0]
        # du.imshow(hal_image, promptdict[saturated_class])
        list_of_hal_imgs0.append(hal_image)
        separate_class_masks0.append(one_class_mask)
        # Save the updated hallucinated image

    # updated_hal_img0 = apply_hallucination_masks_blf(list_of_hal_imgs0, img, separate_class_masks0)

    img_brack1 = reduce_exposure(input_img_to_pipe, 1)
    updated_hal_img0 = apply_hallucination_masks_blf(list_of_hal_imgs0, img_brack1, separate_class_masks0)
    du.imshow(updated_hal_img0, "hal 0")



    # Iteration 2
    # input_img_to_pipe = reduce_exposure(updated_hal_img0, 1)
    input_img_to_pipe = updated_hal_img0
    list_of_hal_imgs1 = []
    separate_class_masks1 = []
    for idx, separate_class_mask in enumerate(separate_class_masks0):
        saturated_mask_left = create_saturated_mask(updated_hal_img0, separate_class_mask)

        if saturated_mask_left is not None:
            # mask_img_to_pipe = saturated_mask_left.astype('float64')
            mask_img_to_pipe = dilate_mask(saturated_mask_left)
            print(saturated_class_indexes[idx], promptdict[saturated_class_indexes[idx]])
            hal_image = pipe(prompt=promptdict[saturated_class_indexes[idx]], image=Image.fromarray(input_img_to_pipe),
                             mask_image=Image.fromarray(mask_img_to_pipe)).images[0]

            # du.imshow(hal_image, promptdict[saturated_class_indexes[idx]])
            list_of_hal_imgs1.append(hal_image)
            separate_class_masks1.append(saturated_mask_left)
        else:
            list_of_hal_imgs1.append(input_img_to_pipe)
            separate_class_masks1.append(None)  # This may need to change from None to a full dark image

    # updated_hal_img1 = apply_hallucination_masks_blf(list_of_hal_imgs1, input_img_to_pipe, separate_class_masks1)

    img_brack2 = reduce_exposure(input_img_to_pipe, 1)
    updated_hal_img1 = apply_hallucination_masks_blf(list_of_hal_imgs1, img_brack2, separate_class_masks1)
    du.imshow(updated_hal_img1, "hal 1")


    # Iteration 3
    # input_img_to_pipe = reduce_exposure(updated_hal_img1, 1)
    input_img_to_pipe = updated_hal_img1
    # cv2.imwrite(f'{out_dir}{fname}_bracket2.png', input_img_to_pipe.astype('uint8'))
    input_img_to_pipe = input_img_to_pipe.astype('uint8')
    list_of_hal_imgs2 = []
    separate_class_masks2 = []
    for idx, separate_class_mask in enumerate(separate_class_masks0):
        saturated_mask_left = create_saturated_mask(updated_hal_img1, separate_class_mask).astype('float64')
        # cv2.imwrite(f'{out_dir}{fname}_3_mask_{idx}.png', saturated_mask_left.astype('uint8'))

        if saturated_mask_left is not None:
            # mask_img_to_pipe = saturated_mask_left
            mask_img_to_pipe = dilate_mask(saturated_mask_left)
            print(saturated_class_indexes[idx], promptdict[saturated_class_indexes[idx]])
            # print(saturated_class, promptdict[saturated_class])
            # du.imshow(Image.fromarray(mask_img_to_pipe), 'entering pipe')
            hal_image = pipe(prompt=promptdict[saturated_class_indexes[idx]], image=Image.fromarray(input_img_to_pipe),
                             mask_image=Image.fromarray(mask_img_to_pipe)).images[0]
            # du.imshow(hal_image, promptdict[saturated_class_indexes[idx]])
            list_of_hal_imgs2.append(hal_image)
            separate_class_masks2.append(saturated_mask_left)
        else:
            list_of_hal_imgs2.append(input_img_to_pipe)
            separate_class_masks2.append(None)  # This may need to change from None to a full dark image
    # updated_hal_img2 = apply_hallucination_masks_blf(list_of_hal_imgs2, input_img_to_pipe, separate_class_masks2)
    img_brack3 = reduce_exposure(input_img_to_pipe, 1)
    updated_hal_img2 = apply_hallucination_masks_blf(list_of_hal_imgs2, img_brack3, separate_class_masks2)

    du.imshow(updated_hal_img2, "hal 2")
    cv2.imwrite(f'{out_dir}{fname}_rsz.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f'{out_dir}{fname}_0.png', cv2.cvtColor(updated_hal_img0, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f'{out_dir}{fname}_1.png', cv2.cvtColor(updated_hal_img1, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f'{out_dir}{fname}_2.png', cv2.cvtColor(updated_hal_img2, cv2.COLOR_BGR2RGB))
    # cv2.imwrite(f'{out_dir}{fname}_mask0.png', updated_hal_img0)
    # cv2.imwrite(f'{out_dir}{fname}_mask1.png', updated_hal_img1)
    # cv2.imwrite(f'{out_dir}{fname}_mask2.png', updated_hal_img2)


    # Save hauginated images.


if __name__ == "__main__":
    promptdict = {
        1: "blue sky with clouds ",
        2: "rocks and sand",
        3: "vegetation and greenery",
        4: "water and waves",
        5: "human being",
        6: "inanimate object",
        7: "road",
        8: "colourful stained glass window"
    }
    bin_label = {
        1: [3],
        2: [14, 17, 35, 47, 69, 92, 95],
        3: [5, 10, 18, 30, 33, 67, 73, 126],
        4: [22, 27, 61, 110, 114, 129],
        5: [13, 127],
        6: [21, 77, 81, 84, 91, 103, 104, 105, 115, 117, 128, 133],
        7: [2, 7, 12, 26, 39, 43, 44, 49, 52, 53, 54, 55, 60, 62, 80, 85, 94, 97, 101, 102, 107, 116, 122, 137],
        8: [1, 4, 6, 8, 9, 11, 15, 16, 19, 20, 23, 24, 25, 28, 29, 31, 32, 34, 36, 40, 41, 42, 45, 46, 48, 50, 51, 56,
            57,
            58, 59, 65, 71, 72, 74, 76, 82, 89, 98, 100, 111, 146],
        9: [37, 38, 63, 64, 66, 68, 70, 75, 78, 79, 83, 86, 87, 88, 90, 93, 96, 99, 106, 108, 109, 112, 113, 118, 119,
            120,
            121, 123, 124, 125, 130, 131, 132, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149,
            150]
    }

    lookuptable = np.zeros(151, dtype='uint8')
    for key, val in bin_label.items():
        for i in range(1, 151):
            if i in val:
                lookuptable[i] = key

    rgb_dir = "./image resized/"
    seg_dir = "./seg_mask/"
    out_dir = "./hal_rgb/"

    # rgb_dir = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/image_resized/"
    # mask_dir = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/bin_mask/"
    # seg_dir = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/seg_mask/"
    # out_dir = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/hal_rgb/"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float32,
    )
    pipe = pipe.to("cuda")

    fname = "a0636"
    main(pipe, promptdict, lookuptable, fname, rgb_dir, seg_dir, out_dir)