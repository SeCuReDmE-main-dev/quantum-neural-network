# IIS Validation Script for Windows 11 Pro
Write-Host "Validating IIS Setup for Windows 11..." -ForegroundColor Cyan

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
Write-Host "Running as Administrator: $isAdmin"
if (-not $isAdmin) {
    Write-Host @"
ERROR: Script must be run as Administrator. Please:
1. Open PowerShell as Administrator
2. Navigate to: $PSScriptRoot
3. Run: .\validate_iis_setup.ps1
"@ -ForegroundColor Red
    exit 1
}

# Check Windows version
$osInfo = Get-CimInstance -ClassName Win32_OperatingSystem
Write-Host "OS Version: $($osInfo.Caption)"

# Check IIS Installation using DISM
Write-Host "`nChecking IIS components..." -ForegroundColor Cyan
$iisComponents = @(
    "IIS-WebServerRole",
    "IIS-WebServer",
    "IIS-CommonHttpFeatures",
    "IIS-ManagementConsole",
    "IIS-ManagementScriptingTools",
    "IIS-ApplicationDevelopment",
    "IIS-NetFxExtensibility45",
    "IIS-ISAPIExtensions",
    "IIS-ISAPIFilter",
    "IIS-ASPNET45",
    "IIS-ApplicationInit",
    "IIS-WindowsAuthentication",
    "IIS-DigestAuthentication",
    "IIS-BasicAuthentication"
)

$missingComponents = @()
foreach ($component in $iisComponents) {
    $status = dism /online /get-featureinfo /featurename:$component
    $installed = $status | Select-String "State : Enabled"
    Write-Host "$component : $(if ($installed) { 'Installed' } else { 'Missing' })"
    if (-not $installed) {
        $missingComponents += $component
    }
}

# Check Web Platform Installer
Write-Host "`nChecking Web Platform Installer..." -ForegroundColor Cyan
$webPiPath = "${env:ProgramFiles}\Microsoft\Web Platform Installer\WebpiCmd.exe"
$webPiInstalled = Test-Path $webPiPath
Write-Host "Web Platform Installer: $(if ($webPiInstalled) { 'Installed' } else { 'Not Installed' })"

# Check port availability
$ports = @(8080, 8081, 8082, 8083)
Write-Host "`nChecking port availability..." -ForegroundColor Cyan
foreach ($port in $ports) {
    $inUse = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    Write-Host "Port $port : $(if ($inUse) { 'In Use' } else { 'Available' })"
}

# Check for Windows Hosting Bundle
Write-Host "`nChecking .NET Core Hosting Bundle..." -ForegroundColor Cyan
$hostingBundleKey = Get-ItemProperty -Path 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Updates\.NET Core*' -ErrorAction SilentlyContinue
Write-Host ".NET Core Hosting Bundle: $(if ($hostingBundleKey) { 'Installed' } else { 'Not Found' })"

# Summary and Action Items
Write-Host "`nValidation Summary:" -ForegroundColor Green

if ($missingComponents.Count -gt 0) {
    Write-Host "`nRequired IIS Components to Install:" -ForegroundColor Yellow
    Write-Host "Run the following commands as Administrator:"
    Write-Host "`nDISM commands:" -ForegroundColor Cyan
    foreach ($component in $missingComponents) {
        Write-Host "dism /online /enable-feature /featurename:$component /all"
    }
}

if (-not $webPiInstalled) {
    Write-Host "`nWeb Platform Installer needs to be installed:" -ForegroundColor Yellow
    Write-Host "Download from: https://www.microsoft.com/web/downloads/platform.aspx"
}

if (-not $hostingBundleKey) {
    Write-Host "`n.NET Core Hosting Bundle needs to be installed:" -ForegroundColor Yellow
    Write-Host "Download from: https://dotnet.microsoft.com/download/dotnet/thank-you/runtime-aspnetcore-6.0.0-windows-hosting-bundle-installer"
}

# Create installation script for missing components
if ($missingComponents.Count -gt 0) {
    $installScript = @"
# Run this script as Administrator to install missing IIS components
Write-Host "Installing IIS components..." -ForegroundColor Cyan
$(($missingComponents | ForEach-Object { "dism /online /enable-feature /featurename:$_ /all" }) -join "`n")

Write-Host "`nInstallation complete. Please run validate_iis_setup.ps1 again to verify." -ForegroundColor Green
"@
    
    $installScriptPath = Join-Path $PSScriptRoot "install_iis_components.ps1"
    Set-Content -Path $installScriptPath -Value $installScript
    Write-Host "`nCreated installation script: $installScriptPath" -ForegroundColor Cyan
}

# Final status
$readyToInstall = $isAdmin -and ($missingComponents.Count -eq 0) -and $webPiInstalled
Write-Host "`nSystem $(if ($readyToInstall) { 'IS' } else { 'IS NOT' }) ready for IIS configuration." -ForegroundColor $(if ($readyToInstall) { 'Green' } else { 'Red' })
if (-not $readyToInstall) {
    Write-Host "Please address the items above before proceeding with IIS configuration." -ForegroundColor Yellow
}