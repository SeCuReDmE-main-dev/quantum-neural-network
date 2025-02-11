# Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

function Install-WFastCGI {
    $pythonPath = "C:\Users\jeans\OneDrive\Desktop\letdo it\AdvancedMaterialScience\venv\Scripts\python.exe"
    $sitePaths = @(
        "C:\inetpub\wwwroot\PersonaManager",
        "C:\inetpub\wwwroot\QuantumTensor",
        "C:\inetpub\wwwroot\MindsDBBridge",
        "C:\inetpub\wwwroot\NeuralNetworkCore"
    )

    Write-Host "Installing and configuring FastCGI..." -ForegroundColor Cyan

    # Install wfastcgi if not already installed
    & $pythonPath -m pip install wfastcgi

    # Create directories if they don't exist
    foreach ($path in $sitePaths) {
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force
        }
    }

    # Copy web.config to each site
    $webConfigSource = Join-Path $PSScriptRoot "web.config"
    foreach ($path in $sitePaths) {
        Copy-Item -Path $webConfigSource -Destination (Join-Path $path "web.config") -Force
    }

    # Reset IIS to apply changes
    iisreset /restart

    Write-Host "`nFastCGI setup complete. Please verify the following:" -ForegroundColor Green
    Write-Host "1. Check IIS Manager to ensure FastCGI handler is configured"
    Write-Host "2. Test endpoints are responding"
    Write-Host "3. Review application pool settings"
}

# Run the installation
Install-WFastCGI