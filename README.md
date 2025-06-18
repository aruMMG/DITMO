# DITMO - Diffusion-based Inverse Tone Mapping
**DITMO** is a novel method for inverse tone mapping that recovers lost details in overexposed SDR images by leveraging semantic-aware diffusion-based inpainting. It introduces two key innovations: (1) using semantic segmentation to guide masked region inpainting, and (2) lifting inpainted SDR content to HDR using a formulation inspired by bracketing techniques. DITMO achieves strong performance on both objective and perceptual metrics across multiple datasets.

![Overview](https://github.com/aruMMG/DITMO/blob/main/assets/Overview.jpg?raw=true)

## Results

### Stable Diffusion Inpainting

Results from DITMO using Stable Diffusion-based inpainting over multiple exposures:
Results from DITMO pipeline with inpainting from the Stable Diffusion model over a range of images. The figures showing details at multiple exposure. 
![Result_SD](https://github.com/aruMMG/DITMO/blob/main/assets/result_SD.png?raw=true)

### ControlNet Inpainting
Results using ControlNet-guided inpainting:
![Result_CN](https://github.com/aruMMG/DITMO/blob/main/assets/result_CN.png?raw=true)


## Getting started
HDR image generation involves two stages:
1. Generating a segmentation mask.

2. Inpainting and HDR reconstruction.

### Creating Segmentation Mask

In order to generate the segmentation mask, we utilized FastFCN. For detailed requirements and instructions, please refer to the [FastFCN repository](https://github.com/wuhuikai/FastFCN).

To create the segmentation mask for your project, follow these steps:

1. Clone or download the FastFCN project from the [FastFCN repository](https://github.com/wuhuikai/FastFCN).

2. Copy the `segment.sh` script provided in this project to the FastFCN directory.

3. Run the following command in your terminal, replacing `[input_path]`, `[output_path]`, and `[weight_file]` with your specific file paths and weight file:
   <code> bash segment.sh [input_path] [output_path] [weight_file] </code>

### HDR image generation
DITMO uses inpainting models from the diffusers library. Make sure it's installed:

```bash
pip install diffusers
```

To generate HDR images using ControlNet and hybrid diffusion:

```bash
python ditmo_inpaint_CN_hybrid_automate.py
```

This will produce multiple HDR outputs — you may choose the most visually suitable one. For single-image generation, refer to controlNet.py.

