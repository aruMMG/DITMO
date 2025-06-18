import cv2
import utils.ditmo_utils as du
from matplotlib import pyplot as plt
def show_array(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()
from PIL import Image
import torch
from VariableParams import *
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.util import img_as_ubyte
import numpy as np

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


def main(pipe, input_img_to_pipe, saturated_class_indexes, single_thres_mask_with_sep_class):
    # Only use first class for inpainting
    saturated_class = saturated_class_indexes[0]
    one_class_mask_untouched = du.prune_mask(single_thres_mask_with_sep_class, saturated_class)

    # Errosion and Dialation
    alpha_one_class_mask = du.createmattes_py(input_img_to_pipe, one_class_mask_untouched, border_t=5)
    refined_alpha = alpha_one_class_mask.copy()
    refined_alpha = np.where(refined_alpha < .98, 0, 1) * 255.0
    eroded = erosion(img_as_ubyte(refined_alpha.astype('uint8')), pair1[0])
    one_class_mask = dilation(eroded, pair1[1])
    mask_img_to_pipe = one_class_mask

    prompt = "Sky with clouds"
    output = inpainting(pipe, input_img_to_pipe, mask_img_to_pipe, prompt)
    output.save("out.png")

    # expand inapinting
    inverse_mask = cv2.bitwise_not(mask_img_to_pipe)
    output1 = inpainting(pipe, np.array(output), inverse_mask, prompt)
    output1.save("out1.png")

# ===================================== Second class inpainting =========================
    # Only use first class for inpainting
    saturated_class = saturated_class_indexes[1]
    one_class_mask_untouched = du.prune_mask(single_thres_mask_with_sep_class, saturated_class)

    # Errosion and Dialation
    alpha_one_class_mask = du.createmattes_py(np.array(output), one_class_mask_untouched, border_t=5)
    refined_alpha = alpha_one_class_mask.copy()
    refined_alpha = np.where(refined_alpha < .98, 0, 1) * 255.0
    eroded = erosion(img_as_ubyte(refined_alpha.astype('uint8')), pair1[0])
    one_class_mask = dilation(eroded, pair1[1])
    mask_img_to_pipe = one_class_mask

    prompt = "Water with reflection of sky"
    output3 = inpainting(pipe, np.array(output), mask_img_to_pipe, prompt)
    output3.save("out3.png")

    # expand inapinting
    inverse_mask = cv2.bitwise_not(mask_img_to_pipe)
    output4 = inpainting(pipe, np.array(output3), inverse_mask, prompt)
    output4.save("out4.png")


if __name__=="__main__":
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

    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # Read image
    img_orig = cv2.imread("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/AutomateDITMO/paper_img/a0418.png")
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img, pad = du.resize_and_pad_image(img_orig)
    img = du.pil_to_opencv(img)
    saturated_mask = du.create_saturated_mask_stage0(img, saturation_threshold=240)
    input_img_to_pipe = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read segmentation mask
    seg_im = cv2.imread("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/itmo/AutomateDITMO/mask/a0418.png")[:,:,0]
    im_label = lookuptable[seg_im + 1]
    im_label, _ = du.resize_and_pad_image(im_label)
    im_label = du.pil_to_opencv(im_label)

    single_thres_mask_with_sep_class = cv2.bitwise_and(im_label, im_label, mask=saturated_mask)
    saturated_class_indexes = du.get_semantic_inpainting_mask_list(single_thres_mask_with_sep_class)
    main(pipe, input_img_to_pipe, saturated_class_indexes, single_thres_mask_with_sep_class)

