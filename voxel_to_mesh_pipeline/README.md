# Voxel-to-Mesh Pipeline

## Overview

This pipeline converts 3D voxel representations to high-quality meshes using neural occupancy networks. It includes:

- **Pretrained Models**: Ready-to-use models for immediate mesh generation
- **Modern Dependencies**: Updated to work with Python 3.9+ and PyTorch 2.0+
- **Windows Compatible**: Full Windows support with PowerShell scripts
- **Standalone**: No dependency on the original codebase

## Features

- Convert voxel grids (.binvox format) to meshes (.off, .ply formats)
- Support for various voxel resolutions (32³, 64³, 128³)
- Batch processing capabilities
- Configurable mesh generation parameters
- Automatic model downloading and setup

## Quick Start

### Option A: Automated Setup (Windows)

```powershell
# Run the automated setup script
.\scripts\setup.ps1
```

### Option B: Manual Setup

#### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate voxel_to_mesh

# OR use pip with requirements.txt
pip install -r requirements.txt

# Install the package
python setup.py build_ext --inplace
```

#### 2. Download Data and Models

```bash
# Cross-platform Python script (recommended)
python scripts/download_data.py

# Windows (PowerShell)
.\scripts\download_data.ps1

# Linux/Mac
bash scripts/download_data.sh
```

#### 3. Test Installation

```bash
# Run test suite to verify everything works
python test_pipeline.py
```

#### 4. Generate Meshes

```bash
# Generate meshes from voxel input using pretrained model
python generate.py configs/voxels/onet_pretrained.yaml

# Process a single voxel file
python generate.py configs/voxels/onet_pretrained.yaml --input-file path/to/your/voxel.binvox
```

## Directory Structure

```text
voxel_to_mesh_pipeline/
├── configs/                 # Configuration files
│   ├── default.yaml        # Default configuration
│   └── voxels/             # Voxel-specific configs
│       ├── onet.yaml       # Base voxel config
│       └── onet_pretrained.yaml  # Pretrained model config
├── models/                 # Model implementations
│   ├── encoder/            # Voxel encoders
│   ├── decoder.py          # Occupancy decoders
│   ├── onet.py            # Occupancy network
│   └── generation.py      # Mesh generation
├── utils/                  # Utility functions
│   ├── voxels.py          # Voxel processing
│   └── common.py          # Common utilities
├── data/                   # Data directory (created after download)
│   └── ShapeNet/          # Downloaded dataset
├── scripts/                # Setup and utility scripts
│   ├── setup.ps1          # Windows automated setup
│   ├── download_data.py   # Cross-platform data download
│   ├── download_data.ps1  # Windows data download
│   └── download_data.sh   # Linux/Mac data download
├── generate.py             # Main generation script
├── test_pipeline.py        # Test suite
├── config.py              # Configuration management
├── setup.py               # Package setup
├── environment.yaml        # Conda environment
└── requirements.txt        # Pip requirements
```

## Configuration

The pipeline uses YAML configuration files with inheritance. Key parameters:

- `method`: "onet" (Occupancy Networks)
- `data.input_type`: "voxels"
- `data.voxels_file`: Input voxel file name (default: "model.binvox")
- `model.encoder`: "voxel_simple" (3D CNN encoder)
- `model.decoder`: "cbatchnorm" (Conditional batch norm decoder)
- `test.threshold`: Occupancy threshold for mesh extraction (default: 0.2)
- `generation.resolution_0`: Base resolution for mesh extraction (default: 32)

## System Requirements

- **Python**: 3.9+ (recommended: 3.10)
- **PyTorch**: 2.0+ with CUDA support (optional but recommended)
- **Memory**: 8GB+ RAM (16GB+ recommended for large models)
- **Storage**: 2GB+ free space for dataset
- **OS**: Windows 10+, Linux, or macOS

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   python setup.py build_ext --inplace
   ```

2. **CUDA Issues**: If you don't have CUDA, the pipeline will run on CPU
   ```bash
   python generate.py configs/voxels/onet_pretrained.yaml --no-cuda
   ```

3. **Extension Build Failures**: Some extensions are optional
   ```bash
   # Install Cython first
   pip install cython
   python setup.py build_ext --inplace
   ```

4. **Download Failures**: Try the Python download script
   ```bash
   python scripts/download_data.py
   ```

## License

Based on the original Occupancy Networks project. See LICENSE for details.

## Citation

If you use this pipeline, please cite the original paper:

```bibtex
@inproceedings{Occupancy Networks,
    title = {Occupancy Networks: Learning 3D Reconstruction in Function Space},
    author = {Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```
