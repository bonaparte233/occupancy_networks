try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
import numpy
import sys
import os

# Get the numpy include directory
numpy_include_dir = numpy.get_include()

# Check if Cython is available
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("Warning: Cython not found. Some extensions will not be available.")

# Extensions list
ext_modules = []

if USE_CYTHON:
    # Marching cubes extension (essential for mesh extraction)
    mcubes_module = Extension(
        'utils.libmcubes.mcubes',
        sources=[
            'utils/libmcubes/mcubes.pyx',
            'utils/libmcubes/pywrapper.cpp',
            'utils/libmcubes/marchingcubes.cpp'
        ],
        language='c++',
        extra_compile_args=['-std=c++11'] if sys.platform != 'win32' else ['/std:c++11'],
        include_dirs=[numpy_include_dir]
    )
    ext_modules.append(mcubes_module)

    # Voxelization extension (for mesh to voxel conversion)
    voxelize_module = Extension(
        'utils.libvoxelize.voxelize',
        sources=[
            'utils/libvoxelize/voxelize.pyx'
        ],
        libraries=['m'] if sys.platform != 'win32' else [],
        include_dirs=[numpy_include_dir]
    )
    ext_modules.append(voxelize_module)

    # Mesh utilities extension
    triangle_hash_module = Extension(
        'utils.libmesh.triangle_hash',
        sources=[
            'utils/libmesh/triangle_hash.pyx'
        ],
        libraries=['m'] if sys.platform != 'win32' else [],
        include_dirs=[numpy_include_dir]
    )
    ext_modules.append(triangle_hash_module)

    # MISE extension (for efficient mesh extraction)
    mise_module = Extension(
        'utils.libmise.mise',
        sources=[
            'utils/libmise/mise.pyx'
        ],
        include_dirs=[numpy_include_dir]
    )
    ext_modules.append(mise_module)

    # Mesh simplification extension
    simplify_mesh_module = Extension(
        'utils.libsimplify.simplify_mesh',
        sources=[
            'utils/libsimplify/simplify_mesh.pyx'
        ],
        include_dirs=[numpy_include_dir]
    )
    ext_modules.append(simplify_mesh_module)

# Setup configuration
setup(
    name='voxel_to_mesh_pipeline',
    version='1.0.0',
    description='Voxel-to-Mesh Pipeline using Occupancy Networks',
    author='Extracted from Occupancy Networks',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-image>=0.20.0',
        'pandas>=2.0.0',
        'h5py>=3.9.0',
        'pyyaml>=6.0',
        'matplotlib>=3.7.0',
        'imageio>=2.31.0',
        'pillow>=9.5.0',
        'trimesh>=3.22.0',
        'tqdm>=4.65.0',
        'tensorboard>=2.13.0',
        'plyfile>=0.7.4',
        'binvox-rw>=1.0.1',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'cython>=0.29.36',
        ],
        'gpu': [
            'torch[cuda]>=2.0.0',
        ],
        'full': [
            'open3d>=0.17.0',
            'pyembree>=0.1.6',
        ]
    },
    ext_modules=cythonize(ext_modules) if USE_CYTHON and ext_modules else [],
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
)

# Post-installation message
print("\n" + "="*60)
print("Voxel-to-Mesh Pipeline Installation")
print("="*60)

if not USE_CYTHON:
    print("⚠️  WARNING: Cython not found!")
    print("   Some extensions were not built. Install Cython for full functionality:")
    print("   pip install cython")
    print("   Then re-run: python setup.py build_ext --inplace")

if ext_modules:
    print("✅ Extensions built successfully!")
    print("   Available extensions:")
    for ext in ext_modules:
        print(f"   - {ext.name}")

print("\n📖 Quick Start:")
print("   1. Download data: python scripts/download_data.py")
print("   2. Generate meshes: python generate.py configs/voxels/onet_pretrained.yaml")
print("   3. Check results in: out/voxels/onet/pretrained/")

print("\n🔗 For more information, see README.md")
print("="*60)
