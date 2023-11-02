import cv2
import torch
from diffusers import StableDiffusionInpaintPipeline
import PIL.Image as Image
from matplotlib import pyplot as plt
import cv2


# device = "cuda"
im_addr = "/home/goswam_a@WMGDS.WMG.WARWICK.AC.UK/stable-diffusion/data/fivek/image resized/"
mask_addr = "/home/goswam_a@WMGDS.WMG.WARWICK.AC.UK/stable-diffusion/data/fivek/bin_mask/"
save_hal_addr = "/home/goswam_a@WMGDS.WMG.WARWICK.AC.UK/stable-diffusion/data/fivek/hal_rgb/"


def imshow(im):
    plt.imshow(im)
    plt.show()


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float32,
)
pipe = pipe.to("cuda")
prompt = "clear sky and with reflection on water"

# prompt = "plane flying in the sky"
# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is

image = Image.open(f"{im_addr}a0418_rsz.png")
# image = Image.open(f"/home/agoswami/PycharmProjects/diffusioninpainting/stable-diffusion/data/inpainting_examples/photo-1583445095369-9c651e7e5d34.png")

imshow(image)
mask_image = Image.open(f"{mask_addr}a0418_binmask.png")
# mask_image = Image.open(f"/home/agoswami/PycharmProjects/diffusioninpainting/stable-diffusion/data/inpainting_examples/photo-1583445095369-9c651e7e5d34_mask.png")

imshow(mask_image)
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
imshow(image)
# image.save(f"{save_hal_addr}/test_{prompt}.png")


# input = cv2.imread(f"{addr}/valley.png")
# dim = (512, 512)
# input = cv2.resize(input, dim, interpolation= cv2.INTER_AREA)
# cv2.imwrite(f'{addr}/valley_sq.png', input)
#
# gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
# imshow(gray)
# mask = cv2.inRange(gray, 210, 255)
# cv2.imwrite(f'{addr}/valley_mask_210.png', mask)
# # imshow(mask)