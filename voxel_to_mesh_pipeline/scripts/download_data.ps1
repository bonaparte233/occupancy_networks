# PowerShell script to download data for voxel-to-mesh pipeline
# Windows-compatible version of download_data.sh

Write-Host "Downloading voxel-to-mesh pipeline data..." -ForegroundColor Green

# Create data directory
$dataDir = "data"
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir
    Write-Host "Created data directory: $dataDir" -ForegroundColor Yellow
}

Set-Location $dataDir

# Download dataset
$datasetUrl = "https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip"
$zipFile = "dataset_small_v1.1.zip"

if (-not (Test-Path $zipFile)) {
    Write-Host "Downloading dataset from $datasetUrl..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri $datasetUrl -OutFile $zipFile -UseBasicParsing
        Write-Host "Dataset downloaded successfully!" -ForegroundColor Green
    }
    catch {
        Write-Error "Failed to download dataset: $_"
        exit 1
    }
} else {
    Write-Host "Dataset already exists: $zipFile" -ForegroundColor Yellow
}

# Extract dataset
if (Test-Path $zipFile) {
    Write-Host "Extracting dataset..." -ForegroundColor Yellow
    try {
        Expand-Archive -Path $zipFile -DestinationPath . -Force
        Write-Host "Dataset extracted successfully!" -ForegroundColor Green
    }
    catch {
        Write-Error "Failed to extract dataset: $_"
        exit 1
    }
} else {
    Write-Error "Dataset file not found: $zipFile"
    exit 1
}

# Copy metadata if needed
$metadataFile = "metadata.yaml"
$shapeNetDir = "ShapeNet"
$targetMetadata = Join-Path $shapeNetDir $metadataFile

if ((Test-Path $metadataFile) -and (-not (Test-Path $targetMetadata))) {
    Write-Host "Copying metadata to ShapeNet directory..." -ForegroundColor Yellow
    Copy-Item $metadataFile $targetMetadata
    Write-Host "Metadata copied successfully!" -ForegroundColor Green
}

# Return to parent directory
Set-Location ..

Write-Host "Data download and setup completed!" -ForegroundColor Green
Write-Host "Dataset location: $dataDir\$shapeNetDir" -ForegroundColor Cyan

# Verify installation
if (Test-Path (Join-Path $dataDir $shapeNetDir)) {
    Write-Host "✓ Dataset installation verified" -ForegroundColor Green
} else {
    Write-Warning "⚠ Dataset installation may have failed"
}
