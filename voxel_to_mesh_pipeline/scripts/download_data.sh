#!/bin/bash
# Shell script to download data for voxel-to-mesh pipeline
# Linux/Mac compatible version

echo "Downloading voxel-to-mesh pipeline data..."

# Create data directory
mkdir -p data
cd data

# Download dataset
DATASET_URL="https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip"
ZIP_FILE="dataset_small_v1.1.zip"

if [ ! -f "$ZIP_FILE" ]; then
    echo "Downloading dataset from $DATASET_URL..."
    if command -v wget &> /dev/null; then
        wget "$DATASET_URL"
    elif command -v curl &> /dev/null; then
        curl -O "$DATASET_URL"
    else
        echo "Error: Neither wget nor curl is available. Please install one of them."
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo "Dataset downloaded successfully!"
    else
        echo "Error: Failed to download dataset"
        exit 1
    fi
else
    echo "Dataset already exists: $ZIP_FILE"
fi

# Extract dataset
if [ -f "$ZIP_FILE" ]; then
    echo "Extracting dataset..."
    unzip -o "$ZIP_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Dataset extracted successfully!"
    else
        echo "Error: Failed to extract dataset"
        exit 1
    fi
else
    echo "Error: Dataset file not found: $ZIP_FILE"
    exit 1
fi

# Copy metadata if needed
METADATA_FILE="metadata.yaml"
SHAPENET_DIR="ShapeNet"
TARGET_METADATA="$SHAPENET_DIR/$METADATA_FILE"

if [ -f "$METADATA_FILE" ] && [ ! -f "$TARGET_METADATA" ]; then
    echo "Copying metadata to ShapeNet directory..."
    cp "$METADATA_FILE" "$TARGET_METADATA"
    echo "Metadata copied successfully!"
fi

# Return to parent directory
cd ..

echo "Data download and setup completed!"
echo "Dataset location: data/$SHAPENET_DIR"

# Verify installation
if [ -d "data/$SHAPENET_DIR" ]; then
    echo "✓ Dataset installation verified"
else
    echo "⚠ Dataset installation may have failed"
fi
