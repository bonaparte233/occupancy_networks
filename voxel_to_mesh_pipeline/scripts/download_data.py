#!/usr/bin/env python3
"""
Cross-platform data download script for voxel-to-mesh pipeline.
Downloads the ShapeNet dataset subset used for occupancy networks.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_file(url, filename, chunk_size=8192):
    """Download a file with progress indication."""
    print(f"Downloading {filename} from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(filename, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
                    else:
                        print(f"\rDownloaded: {downloaded} bytes", end='')
            
            print()  # New line after progress
            return True
            
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        return False


def extract_zip(zip_path, extract_to='.'):
    """Extract a zip file."""
    print(f"Extracting {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def main():
    """Main download function."""
    print("Voxel-to-Mesh Pipeline Data Download")
    print("=" * 50)
    
    # Configuration
    data_dir = Path("data")
    dataset_url = "https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip"
    zip_filename = "dataset_small_v1.1.zip"
    shapenet_dir = data_dir / "ShapeNet"
    metadata_file = "metadata.yaml"
    
    # Create data directory
    data_dir.mkdir(exist_ok=True)
    print(f"Created/verified data directory: {data_dir}")
    
    # Change to data directory
    original_cwd = os.getcwd()
    os.chdir(data_dir)
    
    try:
        # Download dataset if not exists
        if not Path(zip_filename).exists():
            success = download_file(dataset_url, zip_filename)
            if not success:
                print("Failed to download dataset!")
                return False
        else:
            print(f"Dataset already exists: {zip_filename}")
        
        # Extract dataset
        if Path(zip_filename).exists():
            success = extract_zip(zip_filename)
            if not success:
                print("Failed to extract dataset!")
                return False
        else:
            print(f"Dataset file not found: {zip_filename}")
            return False
        
        # Copy metadata if needed
        if Path(metadata_file).exists() and not (shapenet_dir / metadata_file).exists():
            print("Copying metadata to ShapeNet directory...")
            shapenet_dir.mkdir(exist_ok=True)
            shutil.copy2(metadata_file, shapenet_dir / metadata_file)
            print("Metadata copied successfully!")
        
        # Verify installation
        if shapenet_dir.exists():
            print("✓ Dataset installation verified")
            
            # Count categories and models
            try:
                categories = [d for d in shapenet_dir.iterdir() if d.is_dir()]
                total_models = 0
                for cat_dir in categories:
                    models = [d for d in cat_dir.iterdir() if d.is_dir()]
                    total_models += len(models)
                
                print(f"Found {len(categories)} categories with {total_models} total models")
                
            except Exception as e:
                print(f"Warning: Could not count models: {e}")
        else:
            print("⚠ Dataset installation may have failed")
            return False
        
        print("\nData download and setup completed successfully!")
        print(f"Dataset location: {data_dir.absolute() / 'ShapeNet'}")
        
        return True
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
        
    finally:
        # Return to original directory
        os.chdir(original_cwd)


if __name__ == '__main__':
    success = main()
    
    if success:
        print("\n🎉 Ready to generate meshes!")
        print("Next steps:")
        print("1. python generate.py configs/voxels/onet_pretrained.yaml")
        print("2. Check results in: out/voxels/onet/pretrained/")
        sys.exit(0)
    else:
        print("\n❌ Download failed. Please check your internet connection and try again.")
        sys.exit(1)
