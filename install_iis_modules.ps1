# Pre-installation module checker and installer for IIS
Write-Host "Checking and installing required PowerShell modules..." -ForegroundColor Cyan

# List of required modules
$requiredModules = @(
    "WebAdministration",
    "IISAdministration"
)

# Function to safely install module
function Install-RequiredModule {
    param(
        [string]$ModuleName
    )
    
    try {
        if (-not (Get-Module -ListAvailable -Name $ModuleName)) {
            Write-Host "Installing $ModuleName module..." -ForegroundColor Yellow
            # Try installing from PSGallery first
            Install-Module -Name $ModuleName -Force -AllowClobber -Scope CurrentUser -ErrorAction Stop
        } else {
            Write-Host "$ModuleName module is already installed." -ForegroundColor Green
        }
        
        # Import the module
        Import-Module -Name $ModuleName -ErrorAction Stop
        Write-Host "$ModuleName module imported successfully." -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Error installing/importing $ModuleName module: $_" -ForegroundColor Red
        return $false
    }
}

# Check PowerShellGet version
$psGetVersion = (Get-Module -ListAvailable -Name PowerShellGet).Version.Major | Sort-Object -Descending | Select-Object -First 1
if ($psGetVersion -lt 2) {
    Write-Host "Updating PowerShellGet..." -ForegroundColor Yellow
    Install-Module -Name PowerShellGet -Force -AllowClobber -Scope CurrentUser
}

# Install and import required modules
$success = $true
foreach ($module in $requiredModules) {
    if (-not (Install-RequiredModule -ModuleName $module)) {
        $success = $false
    }
}

# Check TLS version
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# Create IIS PowerShell drive if not exists
if ($success) {
    try {
        if (-not (Get-PSDrive -Name IIS -ErrorAction SilentlyContinue)) {
            Write-Host "Creating IIS PowerShell drive..." -ForegroundColor Yellow
            $null = New-PSDrive -Name IIS -PSProvider Registry -Root HKLM:\SOFTWARE\Microsoft\InetStp -ErrorAction Stop
            Write-Host "IIS PowerShell drive created successfully." -ForegroundColor Green
        } else {
            Write-Host "IIS PowerShell drive already exists." -ForegroundColor Green
        }
    }
    catch {
        Write-Host "Error creating IIS PowerShell drive: $_" -ForegroundColor Red
        $success = $false
    }
}

# Final status
if ($success) {
    Write-Host "`nAll required PowerShell modules are installed and configured." -ForegroundColor Green
    Write-Host "You can now proceed with running validate_iis_setup.ps1" -ForegroundColor Cyan
} else {
    Write-Host "`nSome components failed to install. Please check the errors above and retry." -ForegroundColor Red
}