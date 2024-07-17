#!/bin/bash

# Function to prompt the user for input
prompt_user() {
    read -p "Please enter the directory path: " user_path
    export USER_PATH=$user_path
}

# Prompt the user for input
prompt_user

# Copy the user-specified directory to the workspace
cp -r $USER_PATH /workspace/data

# cuda가 가능한지 확인
python3 /workspace/check_cuda.py

# Iterate over each subdirectory in the user-specified directory
for dir in /workspace/data/*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        # Run the NIA_data_splitter.py script for each subdirectory
        python3 /workspace/NIA_data_splitter.py --data_path "$dir/rawData/Car" --label_path "$dir/labelingData/Car" --output_path /workspace/Output/"$dir_name"
    fi
done

# Run the NIA_train.py script
python3 /workspace/NIA_train.py --data_path /workspace/Output/train --val_path /workspace/Output/val

# Run the NIA_test.py script
python3 /workspace/NIA_test.py --test_data_path /workspace/Output/test/ --model_path /workspace/Output/result/best_mask_rcnn_model.pt


# Copy the result directory to the host
cp -r /workspace/Output/result/image /output/


# Keep the container running
exec "$@"
