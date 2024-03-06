#!/bin/bash

# Source and destination folders
source_folder="airbus-vessel-recognition/training_data_1k_256/train/mask"
destination_folder="airbus-vessel-recognition/training_data_1k_256/test/mask"

# Create destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Move the first 100 files from source to destination
counter=0
for file in "$source_folder"/*; do
    if [ $counter -lt 100 ]; then
        mv "$file" "$destination_folder"
        ((counter++))
    else
        break
    fi
done


# Source and destination folders
source_folder="airbus-vessel-recognition/training_data_1k_256/train/img"
destination_folder="airbus-vessel-recognition/training_data_1k_256/test/img"

# Create destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Move the first 100 files from source to destination
counter=0
for file in "$source_folder"/*; do
    if [ $counter -lt 100 ]; then
        mv "$file" "$destination_folder"
        ((counter++))
    else
        break
    fi
done