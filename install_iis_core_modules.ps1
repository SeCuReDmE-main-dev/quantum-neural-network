# PowerShell Core (pwsh) script for IIS module installation
Write-Host "Installing IIS modules using PowerShell Core..." -ForegroundColor Cyan

# Check for administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "This script must be run as Administrator. Please run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

# Register Microsoft PowerShell Gallery
Write-Host "Registering PowerShell Gallery..." -ForegroundColor Yellow
if (Get-PSRepository -Name "PSGallery" -ErrorAction SilentlyContinue) {
    Set-PSRepository -Name "PSGallery" -InstallationPolicy Trusted
} else {
    Register-PSRepository -Default
    Set-PSRepository -Name "PSGallery" -InstallationPolicy Trusted
}

# Install PowerShellGet if needed
if (-not (Get-Module -ListAvailable -Name PowerShellGet)) {
    Write-Host "Installing PowerShellGet..." -ForegroundColor Yellow
    Install-Module -Name PowerShellGet -Force -AllowClobber -Scope CurrentUser
}

# Install required modules
$modules = @(
    @{
        Name = "IISAdministration"
        MinimumVersion = "1.1.0.0"
    }
)

foreach ($module in $modules) {
    Write-Host "Installing $($module.Name)..." -ForegroundColor Yellow
    try {
        if (-not (Get-Module -ListAvailable -Name $module.Name)) {
            Install-Module -Name $module.Name -MinimumVersion $module.MinimumVersion -Force -AllowClobber -Scope CurrentUser
        }
        Import-Module -Name $module.Name -Force -ErrorAction Stop
        Write-Host "$($module.Name) installed and imported successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "Error installing $($module.Name): $_" -ForegroundColor Red
    }
}

# Enable IIS Windows Features
$features = @(
    "IIS-WebServerRole",
    "IIS-WebServer",
    "IIS-CommonHttpFeatures",
    "IIS-ManagementConsole",
    "IIS-ManagementScriptingTools",
    "IIS-ManagementService",
    "IIS-IIS6ManagementCompatibility",
    "IIS-Metabase",
    "IIS-WMICompatibility",
    "IIS-LegacySnapIn"
)

foreach ($feature in $features) {
    Write-Host "Enabling Windows Feature: $feature" -ForegroundColor Yellow
    $result = dism /online /enable-feature /featurename:$feature /all
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$feature enabled successfully" -ForegroundColor Green
    } else {
        Write-Host "Error enabling $feature" -ForegroundColor Red
    }
}

# Create IIS PowerShell drive
Write-Host "Creating IIS PowerShell drive..." -ForegroundColor Yellow
try {
    if (-not (Get-PSDrive -Name IIS -ErrorAction SilentlyContinue)) {
        $null = New-PSDrive -Name IIS -PSProvider Registry -Root HKLM:\SOFTWARE\Microsoft\InetStp -Scope Script
        Write-Host "IIS drive created successfully" -ForegroundColor Green
    } else {
        Write-Host "IIS drive already exists" -ForegroundColor Green
    }
}
catch {
    Write-Host "Error creating IIS drive: $_" -ForegroundColor Red
}

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Cyan
$testCommands = @(
    @{
        Command = "Get-IISServerManager"
        Name = "IIS Server Manager"
    },
    @{
        Command = "Get-IISAppPool"
        Name = "IIS Application Pools"
    }
)

foreach ($test in $testCommands) {
    try {
        $null = Invoke-Expression $test.Command
        Write-Host "$($test.Name) command successful" -ForegroundColor Green
    }
    catch {
        Write-Host "$($test.Name) command failed: $_" -ForegroundColor Red
    }
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Install the Web Platform Installer (WebPI) if not already installed"
Write-Host "2. Install URL Rewrite Module using WebPI"
Write-Host "3. Install .NET Core Hosting Bundle"
Write-Host "4. Run 'iisreset' to restart IIS"
Write-Host "5. Run validate_iis_setup.ps1 again"