# PowerShell script to set up the voxel-to-mesh pipeline environment
# Windows-compatible setup script

param(
    [switch]$SkipConda,
    [switch]$SkipData,
    [string]$CondaEnv = "voxel_to_mesh"
)

Write-Host "Setting up Voxel-to-Mesh Pipeline..." -ForegroundColor Green

# Check if conda is available
if (-not $SkipConda) {
    try {
        $condaVersion = conda --version
        Write-Host "Found conda: $condaVersion" -ForegroundColor Yellow
    }
    catch {
        Write-Warning "Conda not found. Please install Anaconda or Miniconda first."
        Write-Host "Download from: https://www.anaconda.com/products/distribution" -ForegroundColor Cyan
        exit 1
    }

    # Create conda environment
    Write-Host "Creating conda environment: $CondaEnv" -ForegroundColor Yellow
    try {
        conda env create -f environment.yaml -n $CondaEnv
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Conda environment created successfully!" -ForegroundColor Green
        }
        else {
            Write-Warning "Environment may already exist. Updating..."
            conda env update -f environment.yaml -n $CondaEnv
        }
    }
    catch {
        Write-Error "Failed to create conda environment: $_"
        exit 1
    }

    # Activate environment and install package
    Write-Host "Activating environment and installing package..." -ForegroundColor Yellow
    try {
        & conda activate $CondaEnv
        python setup.py build_ext --inplace
        Write-Host "✓ Package installed successfully!" -ForegroundColor Green
    }
    catch {
        Write-Warning "Failed to build extensions. Some functionality may be limited."
        Write-Host "You can try building manually after activating the environment:" -ForegroundColor Cyan
        Write-Host "  conda activate $CondaEnv" -ForegroundColor Cyan
        Write-Host "  python setup.py build_ext --inplace" -ForegroundColor Cyan
    }
}

# Download data
if (-not $SkipData) {
    Write-Host "Downloading dataset..." -ForegroundColor Yellow
    try {
        & .\scripts\download_data.ps1
        Write-Host "✓ Dataset downloaded successfully!" -ForegroundColor Green
    }
    catch {
        Write-Warning "Failed to download dataset automatically."
        Write-Host "You can download manually by running:" -ForegroundColor Cyan
        Write-Host "  .\scripts\download_data.ps1" -ForegroundColor Cyan
    }
}

# Create output directories
$outputDirs = @("out", "out\voxels", "out\voxels\onet")
foreach ($dir in $outputDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
        Write-Host "Created directory: $dir" -ForegroundColor Yellow
    }
}

Write-Host "`nSetup completed!" -ForegroundColor Green
Write-Host "To use the pipeline:" -ForegroundColor Cyan
Write-Host "1. Activate the environment: conda activate $CondaEnv" -ForegroundColor White
Write-Host "2. Generate meshes: python generate.py configs\voxels\onet_pretrained.yaml" -ForegroundColor White
Write-Host "3. Check results in: out\voxels\onet\pretrained\" -ForegroundColor White

# Test installation
Write-Host "`nTesting installation..." -ForegroundColor Yellow
try {
    & conda activate $CondaEnv
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import trimesh; print(f'Trimesh version: {trimesh.__version__}')"
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
    Write-Host "✓ Core dependencies verified!" -ForegroundColor Green
}
catch {
    Write-Warning "Some dependencies may not be properly installed."
}

Write-Host "`n🎉 Voxel-to-Mesh Pipeline setup complete!" -ForegroundColor Green
