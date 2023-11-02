# DITMO - Diffusion-based Inverse Tone Mapping

## Creating Segmentation Mask

In order to generate the segmentation mask, we utilized FastFCN. For detailed requirements and instructions, please refer to the [FastFCN repository](https://github.com/wuhuikai/FastFCN).

To create the segmentation mask for your project, follow these steps:

1. Clone or download the FastFCN project from the [FastFCN repository](https://github.com/wuhuikai/FastFCN).

2. Copy the `segment.sh` script provided in this project to the FastFCN directory.

3. Run the following command in your terminal, replacing `[input_path]`, `[output_path]`, and `[weight_file]` with your specific file paths and weight file:
<pre>   
~~~bash
   bash segment.sh [input_path] [output_path] [weight_file]
~~~
</pre>
