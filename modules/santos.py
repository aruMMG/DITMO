import numpy as np
from PIL import Image
import torch.nn as nn
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import imageio
import cv2

from torchvision import transforms
import torch
from utils.ditmo_utils import resize_and_pad_image
from utils.softconvmask import SoftConvNotLearnedMaskUNet

def writeEXR(img, file):
    try:
        # Ensure the image has the correct shape
        img = np.squeeze(img)
        
        # Save the HDR image using imageio
        imageio.imwrite(file, img.astype(np.float32))

    except Exception as e:
        raise IOError(f"Failed writing EXR: {e}")

# Write exposure compensated 8-bit image
def writeLDR(img, file, exposure=0):

    # Convert exposure fstop in linear domain to scaling factor on display values
    sc = np.power(np.power(2.0, exposure), 0.5)

    img = ((sc * img[..., ::-1]) * 255).astype(np.uint8)
    try:
        #scipy.misc.toimage(sc*np.squeeze(img), cmin=0.0, cmax=1.0).save(file)
        cv2.imwrite(file, img)
    except Exception as e:
        raise IOException("Failed writing LDR image: %s"%e)




def load_image(name_jpg):
    return np.asarray(Image.open(name_jpg).convert('RGB')).astype(np.float32)/255.0

def saturated_channel_(im, th):
    return np.minimum(np.maximum(0.0, im - th) / (1 - th), 1)

def get_saturated_regions(im, th=0.95):
    w,h,ch = im.shape

    mask_conv = np.zeros_like(im)
    for i in range(ch):
        mask_conv[:,:,i] = saturated_channel_(im[:,:,i], th)

    return mask_conv#, mask

def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name, map_location='cpu')
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])

    epoch = ckpt_dict['n_iter'] if 'n_iter' in ckpt_dict else 0
    step = ckpt_dict['step'] if 'step' in ckpt_dict else 0

    return step,  epoch

def unnormalize(x, MEAN=[0.485, 0.456, 0.406],STD=[0.229, 0.224, 0.225]):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x

def santos(image, conv_mask, weight="utils/ldr2hdr.pth"):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    conv_mask = torch.from_numpy(conv_mask).permute(2,0,1)

    image = img_transform(image)
    image = image.unsqueeze(0)
    conv_mask = conv_mask.unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SoftConvNotLearnedMaskUNet().to(device)
    model.print_network()
    load_ckpt(weight, [('model', model)])
    print("Starting prediction...\n\n")
    model.eval()
    with torch.no_grad():
            pred_img = model(image.to(device), conv_mask.to(device))

    print("\t santos done...\n")
    print(torch.max(pred_img))
    image = unnormalize(image, MEAN=MEAN, STD=STD).permute(0,2,3,1).numpy()[0,:,:,:]
    mask = conv_mask.permute(0,2,3,1).numpy()[0,:,:,:]
    pred_img = pred_img.cpu().permute(0,2,3,1).numpy()[0,:,:,:]

    y_predict = np.exp(pred_img)-1
    gamma = np.power(image, 2)

    # save EXR images.
    H = mask*gamma + (1-mask)*y_predict
    H = H/np.max(H)
    return H, mask
if __name__=="__main__":





    image_path = "/home/aru/Downloads/check.png"
    image = resize_and_pad_image(image_path)
    # load image
    # image = load_image(image_path)
    image = np.asarray(image.convert('RGB')).astype(np.float32)/255.0
    # get saturation mask
    conv_mask = 1 - get_saturated_regions(image)

    H, mask = santos(image, conv_mask)
    writeEXR(H, "check_folder/img.exr")
    writeLDR(mask, "check_folder/img.png")
