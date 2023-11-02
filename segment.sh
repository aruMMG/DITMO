#!/bin/bash

# Path to the Python script and its arguments
python_script="python -m experiments.segmentation.test_single_image"
args="--dataset ade20k --model encnet --jpu JPU --backbone resnet50"

# Input and output directories
input_dir="$1"
output_dir="$2"
weight_file = "$3"
# Check if input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Input directory '$input_dir' does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

# Process each image in the input directory
for image_path in "$input_dir"/*; do
    if [ -f "$image_path" ]; then
        image_name=$(basename "$image_path")
        output_path="$output_dir/$image_name"
        
        # Run the Python script with appropriate arguments
        eval "$python_script $args --input-path '$image_path' --save-path '$output_path' --resume '$weight_file'"
        
        echo "Processed: $image_name -> $output_path"
    fi
done

echo "Processing complete."
