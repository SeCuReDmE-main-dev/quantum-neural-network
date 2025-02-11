# Requires -RunAsAdministrator

Write-Host "Installing IIS Prerequisites..." -ForegroundColor Cyan

# Function to ensure NuGet is available and registered
function Initialize-PackageProviders {
    Write-Host "Initializing package providers..." -ForegroundColor Yellow
    
    # Install NuGet if not present
    if (-not (Get-PackageProvider -Name NuGet -ErrorAction SilentlyContinue)) {
        try {
            Install-PackageProvider -Name NuGet -Force -Scope CurrentUser
        } catch {
            Write-Error "Failed to install NuGet provider: $_"
            return $false
        }
    }
    
    # Register PSGallery if not registered
    if (-not (Get-PSRepository -Name PSGallery -ErrorAction SilentlyContinue)) {
        try {
            Register-PSRepository -Default -InstallationPolicy Trusted
        } catch {
            Write-Error "Failed to register PSGallery: $_"
            return $false
        }
    }
    
    return $true
}

# Function to install Windows Features
function Install-RequiredWindowsFeatures {
    Write-Host "Installing required Windows features..." -ForegroundColor Yellow
    
    $features = @(
        "Web-Server",
        "Web-WebServer",
        "Web-Common-Http",
        "Web-Mgmt-Tools",
        "Web-Scripting-Tools"
    )
    
    foreach ($feature in $features) {
        try {
            $featureState = dism /online /get-featureinfo /featurename:$feature
            if ($featureState -notmatch "State : Enabled") {
                Write-Host "Installing feature: $feature" -ForegroundColor Yellow
                dism /online /enable-feature /featurename:$feature /all
            } else {
                Write-Host "Feature already installed: $feature" -ForegroundColor Green
            }
        } catch {
            Write-Error "Failed to install feature $feature : $_"
            return $false
        }
    }
    
    return $true
}

# Function to install PowerShell modules
function Install-RequiredModules {
    Write-Host "Installing required PowerShell modules..." -ForegroundColor Yellow
    
    # Remove existing modules if present
    Remove-Module WebAdministration -ErrorAction SilentlyContinue
    Remove-Module IISAdministration -ErrorAction SilentlyContinue
    
    # Install Microsoft.Web.Administration assembly
    $webAdminPath = "${env:ProgramFiles}\Reference Assemblies\Microsoft\IIS"
    if (-not (Test-Path $webAdminPath)) {
        New-Item -Path $webAdminPath -ItemType Directory -Force
    }
    
    try {
        # Install modules from gallery
        Install-Module -Name IISAdministration -Force -AllowClobber -Scope CurrentUser
        
        # Create IIS Drive
        if (-not (Get-PSDrive -Name IIS -ErrorAction SilentlyContinue)) {
            New-PSDrive -Name IIS -PSProvider Registry -Root HKLM:\SOFTWARE\Microsoft\InetStp -ErrorAction Stop
        }
        
        return $true
    } catch {
        Write-Error "Failed to install required modules: $_"
        return $false
    }
}

# Main installation process
$success = $true

# Step 1: Initialize package providers
if (-not (Initialize-PackageProviders)) {
    $success = $false
    Write-Error "Failed to initialize package providers"
}

# Step 2: Install Windows Features
if ($success -and -not (Install-RequiredWindowsFeatures)) {
    $success = $false
    Write-Error "Failed to install Windows features"
}

# Step 3: Install PowerShell modules
if ($success -and -not (Install-RequiredModules)) {
    $success = $false
    Write-Error "Failed to install PowerShell modules"
}

# Final status
if ($success) {
    Write-Host "`nPrerequisites installation completed successfully!" -ForegroundColor Green
    Write-Host "Please restart PowerShell and run validate_iis_setup.ps1 again."
} else {
    Write-Host "`nPrerequisites installation failed. Please check the errors above." -ForegroundColor Red
}

# Create verification script
$verifyScript = @"
# Verify installation
Write-Host "Verifying IIS modules..."
Import-Module WebAdministration -ErrorAction SilentlyContinue
Import-Module IISAdministration -ErrorAction SilentlyContinue

if (Get-Module WebAdministration) {
    Write-Host "WebAdministration module loaded successfully" -ForegroundColor Green
} else {
    Write-Host "WebAdministration module not loaded" -ForegroundColor Red
}

if (Get-Module IISAdministration) {
    Write-Host "IISAdministration module loaded successfully" -ForegroundColor Green
} else {
    Write-Host "IISAdministration module not loaded" -ForegroundColor Red
}

if (Get-IISAppPool) {
    Write-Host "IIS Administration working correctly" -ForegroundColor Green
} else {
    Write-Host "IIS Administration not working correctly" -ForegroundColor Red
}
"@

Set-Content -Path ".\verify_iis_modules.ps1" -Value $verifyScript
Write-Host "`nCreated verification script: verify_iis_modules.ps1"