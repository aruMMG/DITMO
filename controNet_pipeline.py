import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from os.path import isfile, join
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import utils.ditmo_utils as du
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
# from soft_segment import *
from skimage.util import img_as_ubyte
from VariableParams import *


def ours_make_inpaint_condition(image, image_mask):
    image = image.astype(np.float32) / 255.0
    image_mask = image_mask.astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def inpainting(pipe, input_img_to_pipe, mask_img_to_pipe, prompt):

    # Create control image
    control_image = ours_make_inpaint_condition(input_img_to_pipe, mask_img_to_pipe)

    # Run pipe
    output = pipe(
        prompt,
        num_inference_steps=20,
        eta=1.0,
        image=Image.fromarray(input_img_to_pipe),
        mask_image=Image.fromarray(mask_img_to_pipe),
        control_image=control_image,
    ).images[0]
    return output


sat_thresh = 240
exp_t = 400


def main(pipe, promptdict, morph_rad_pair, lookuptable, fname, img_dir, seg_img_dir, save_path):
    img_orig = cv2.imread(f'{img_dir}{fname}')

    # --------------------------------------------- Temp image files
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig_resized = img_orig.copy()
    ori_dim = img_orig.shape
    # ---------------------------------------------

    img, pad = du.resize_and_pad_image(img_orig)
    img = du.pil_to_opencv(img)

    saturated_mask = du.create_saturated_mask_stage0(img, saturation_threshold=sat_thresh)
    seg_im = cv2.imread(f'{seg_img_dir}{fname}')[:, :, 0]
    im_label = lookuptable[seg_im + 1]
    im_label, _ = du.resize_and_pad_image(im_label)
    im_label = du.pil_to_opencv(im_label)
    single_thres_mask_with_sep_class = cv2.bitwise_and(im_label, im_label, mask=saturated_mask)
    saturated_class_indexes = du.get_semantic_inpainting_mask_list(single_thres_mask_with_sep_class)

    input_img_to_pipe = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image2merge = img_orig.copy()
    img_list_ldr = []
    exp_list = []
    base_exp_t = 200
    base_exp = 0
    predicted_brack = 0
    for idx, saturated_class in enumerate(saturated_class_indexes):
        list_of_hal_portions = []
        list_of_soft_masks = []
        one_class_mask_untouched = du.prune_mask(single_thres_mask_with_sep_class, saturated_class)

        # --------------------------------------------- Pre-process - Refine mask by Alpha matting (opt)
        alpha_one_class_mask = du.createmattes_py(input_img_to_pipe, one_class_mask_untouched, border_t=5)
        refined_alpha = alpha_one_class_mask.copy()
        refined_alpha = np.where(refined_alpha < .98, 0, 1) * 255.0
        eroded = erosion(img_as_ubyte(refined_alpha.astype('uint8')), morph_rad_pair[0])

        # ---------------------------------------------- Pre-process - Refine mask by morphological filters
        one_class_mask = dilation(eroded, morph_rad_pair[1])
        # --------------------------------------------------------Inpainting
        mask_img_to_pipe = one_class_mask
        print("...inpainting " + promptdict[saturated_class])

        hal_image = inpainting(pipe, input_img_to_pipe, mask_img_to_pipe, promptdict[saturated_class])
        # -------------------------------------------------------- Just hallucinated mask
        hal_cv = np.array(hal_image)
        hal_unpadded = du.remove_pad(hal_cv, pad)
        hal_unpadded_shape = hal_unpadded.shape
        img_orig_resized = cv2.resize(img_orig_resized, (hal_unpadded_shape[1], hal_unpadded_shape[0]),
                                      interpolation=cv2.INTER_AREA)

        mask_unpadded = du.remove_pad(one_class_mask, pad)
        just_hal = du.get_halucinatedrgb(mask_unpadded, hal_unpadded)

        # ------------------------------------------------------ Inpaint extension mask
        inverse_mask = cv2.bitwise_not(mask_img_to_pipe)
        print("Inpainting extension mask...")
        hal_ext = inpainting(pipe, hal_cv, inverse_mask, promptdict[saturated_class])

        extension_cv = np.array(hal_ext)
        extension_cv_unpadded = du.remove_pad(extension_cv, pad)

        # ------------------------------------------------------------ Prep for merging
        mask_orig = cv2.resize(mask_unpadded, (ori_dim[1], ori_dim[0]), interpolation=cv2.INTER_NEAREST)
        alpha_mask_unpadded_4x = du.createmattes_py(img_orig, mask_orig, border_t=10, scale=.5).astype("float32")
        extension_cv_unpadded_4x = du.upscale_superresx4(extension_cv_unpadded.astype(np.float32) / 255.0)
        extension_cv_unpadded_4x = cv2.resize(extension_cv_unpadded_4x, (ori_dim[1], ori_dim[0]),
                                              interpolation=cv2.INTER_AREA)
        list_of_soft_masks.append(alpha_mask_unpadded_4x)
        list_of_hal_portions.append(extension_cv_unpadded_4x)

        predicted_brack = du.predict_bracket(extension_cv_unpadded, mask_unpadded)
        brack2merge = base_exp - predicted_brack
        base_exp = predicted_brack
        base_image = du.modify_exposure(image2merge, brack2merge)
        merged_hallucination = du.merge(list_of_hal_portions, base_image, list_of_soft_masks)

        image2merge = merged_hallucination * 255.0
        temp_im, _ = du.resize_and_pad_image(image2merge.astype('uint8'))
        input_img_to_pipe = np.array(temp_im)

    # --------------------------------------------- Upscale 4x
    cv2.imwrite(f'{save_path}{suffix_code}_brack_0.png', cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    for fstop in range(predicted_brack):
        im_out = du.modify_exposure(image2merge, fstop)
        cv2.imwrite(f'{save_path}{suffix_code}_brack_{int(predicted_brack - fstop)}.png',
                    cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB))

    for fstop in range(2):
        im_out = du.modify_exposure(image2merge, int(-(fstop + 1)))
        cv2.imwrite(f'{save_path}{suffix_code}_brack_{int(predicted_brack + fstop + 1)}.png',
                    cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB))

    for brack in range(predicted_brack + 3):
        ldr = cv2.imread(f"{save_path}{suffix_code}_brack_{brack}.png")
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
        base_exp_t = base_exp_t / (2 ** brack)
        img_list_ldr.append(ldr)
        exp_list.append(base_exp_t)

    hdr = du.createHDR_multibrack(img_list_ldr, exp_list)
    hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{save_path}{suffix_code}_mergedhdr_multibrack_2k.exr", hdr)

    print(f'Done..file[{k+1}/{len(filenames)}]')
    print('---------------------------------------------------------------')


if __name__ == "__main__":

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

    rgb_dir = '/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/AutomateDITMO/paper_img/'
    seg_dir = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/AutomateDITMO/mask/"
    out_dir = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/AutomateDITMO/outputs/hal/"

    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe = pipe.to("cuda")

    filenames = [f.split('_')[0] for f in os.listdir(rgb_dir) if isfile(join(rgb_dir, f))]
    failed_files = []

    for i, pdict in enumerate(dicts):
        if not os.path.exists(f"{out_dir}{legend_dict[i]}"):
            os.mkdir(f"{out_dir}{legend_dict[i]}")

        for j, morph_pair in enumerate(morphology_radius_pairs):
            if not os.path.exists(f"{out_dir}{legend_dict[i]}/{legend_rad[j]}"):
                os.mkdir(f"{out_dir}{legend_dict[i]}/{legend_rad[j]}")
            print(f'Iteration for {legend_dict[i]}, {legend_rad[j]}')
            for k, fname in enumerate(filenames):
                print(f"Ditmo for -> {fname}")
                if not os.path.exists(f"{out_dir}{legend_dict[i]}/{legend_rad[j]}/{fname}"):
                    os.mkdir(f"{out_dir}{legend_dict[i]}/{legend_rad[j]}/{fname}")
                path = f"{out_dir}{legend_dict[i]}/{legend_rad[j]}/{fname}/"
                suffix_code = f'{legend_dict[i]}_{legend_rad[j]}_{fname}'
                try:
                    main(pipe, pdict, morph_pair, lookuptable, fname, rgb_dir, seg_dir, path)
                except ValueError as ve:
                    failed_files.append(fname)
    with open(f'{out_dir}skippedfiles.npy', 'wb') as f:
        np.save(f, failed_files)
