# Voxel2Mesh

A simple and reliable tool for converting voxel data to 3D meshes using the marching cubes algorithm.

This tool provides a clean, efficient implementation of voxel-to-mesh conversion with comprehensive evaluation and visualization capabilities.

## Features

- **Simple API**: Easy-to-use Python interface for voxel to mesh conversion
- **Reliable Algorithm**: Uses proven marching cubes algorithm for consistent results
- **Multiple Input Formats**: Supports .binvox, .npy, and .npz voxel files
- **Multiple Output Formats**: Supports .off, .ply, .obj and other mesh formats
- **Command Line Interface**: Simple CLI for batch processing
- **No Compilation Required**: Pure Python implementation using scikit-image
- **Evaluation and Visualization**: Complete mesh evaluation metrics and 3D visualizations
- **Organized Output Structure**: Automatic organization of results in data/input and data/output folders
- **Fast Processing**: No GPU required, works efficiently on any system

## Algorithm

This tool uses the **marching cubes algorithm** for voxel-to-mesh conversion, which provides:

- ✅ **Reliable Results**: Consistent and predictable mesh generation
- ✅ **Fast Processing**: No GPU required, works efficiently on any system  
- ✅ **High Quality**: Generates watertight meshes with proper topology
- ✅ **Complete Pipeline**: Includes evaluation metrics and visualizations
- ✅ **No Dependencies**: No need for neural network models or training data

The marching cubes approach is a proven, industry-standard method for surface reconstruction from voxel data.

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone or download this repository
cd voxel2mesh

# Create and activate conda environment
conda env create -f environment.yml
conda activate voxel2mesh
```

### Option 2: Pip Installation

```bash
# Clone or download this repository
cd voxel2mesh

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Data Folder Structure

The tool uses a organized data folder structure:

```text
voxel2mesh/
├── data/
│   ├── input/     # Place your voxel files here
│   └── output/    # Generated meshes and visualizations
```

### Python API

#### Method 1: Using Data Structure (Recommended)

```python
from voxel2mesh import Voxel2Mesh

# Initialize the converter
converter = Voxel2Mesh(threshold=0.5)

# Process voxel file with automatic output organization
result = converter.process_voxel_file('my_voxel.binvox')

# This will:
# - Look for data/input/my_voxel.binvox
# - Generate mesh using marching cubes
# - Save to data/output/my_voxel/
# - Create visualizations (voxels, mesh, comparison)
# - Save evaluation results and statistics
```

#### Method 2: Direct Conversion

```python
# Convert voxel file to mesh
mesh = converter.convert_voxels_to_mesh('input.binvox', 'output.off')

# Or use numpy array directly
import numpy as np
voxels = np.load('voxels.npy')  # Shape: (32, 32, 32) or similar
mesh = converter.convert_voxels_to_mesh(voxels, 'output.ply')

print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
print(f"Watertight: {mesh.is_watertight}")
```

### Command Line Interface

#### Using Data Structure

```bash
# Place your voxel file in data/input/ then run:
python cli.py my_voxel.binvox --verbose

# Custom threshold
python cli.py my_voxel.binvox --threshold 0.3 --verbose
```

#### Traditional Usage

```bash
# Specify exact input and output paths
python cli.py input.binvox --output output.off --verbose
```

## Supported Formats

### Input Formats
- **.binvox**: Binary voxel format (from binvox tool)
- **.npy**: NumPy array files
- **.npz**: Compressed NumPy array files

### Output Formats
- **.off**: Object File Format
- **.ply**: Polygon File Format  
- **.obj**: Wavefront OBJ
- **.stl**: STereoLithography format
- And other formats supported by trimesh

## Configuration

Key parameters:

- `threshold`: Threshold for marching cubes (default: 0.5)

## Examples

### Simple Example

```python
from voxel2mesh import Voxel2Mesh
import numpy as np

# Create test voxel data (sphere)
size = 32
center = size // 2
radius = 8

voxels = np.zeros((size, size, size), dtype=np.float32)
for i in range(size):
    for j in range(size):
        for k in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
            if dist <= radius:
                voxels[i, j, k] = 1.0

# Convert to mesh
converter = Voxel2Mesh()
mesh = converter.convert_voxels_to_mesh(voxels, 'sphere.off')
print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
```

### Batch Processing

```python
import os
from voxel2mesh import Voxel2Mesh

converter = Voxel2Mesh()

# Process all files in data/input/
for filename in os.listdir('data/input'):
    if filename.endswith(('.binvox', '.npy', '.npz')):
        try:
            result = converter.process_voxel_file(filename)
            print(f"Processed {filename} -> {result['output_dir']}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
```

## Output Structure

For each processed file, the tool generates:

- **mesh.off**: Generated 3D mesh
- **voxels.png**: 3D visualization of input voxels
- **mesh.png**: 3D visualization of generated mesh
- **comparison.png**: Side-by-side comparison
- **results.json**: Evaluation metrics and metadata

## Requirements

- Python 3.9+
- NumPy 1.21+
- SciPy 1.7+
- scikit-image 0.19+
- trimesh 3.15+
- matplotlib 3.5+

## Performance

Typical performance on a modern CPU:

- **32³ voxels**: ~0.1 seconds
- **64³ voxels**: ~0.5 seconds  
- **128³ voxels**: ~2 seconds

Results are watertight meshes with proper topology.

## Troubleshooting

### Common Issues

1. **Empty mesh output**: Try adjusting the `threshold` parameter (default: 0.5)
2. **Import errors**: Make sure all dependencies are installed correctly
3. **File not found**: Check that input files are in the correct location

### Getting Help

If you encounter issues:

1. Check that your input voxel data is in the correct format
2. Verify that all dependencies are installed
3. Try with different threshold values (0.1 to 0.9)
4. Use `--verbose` flag for detailed output

## License

This project is provided as-is for research and educational purposes.
