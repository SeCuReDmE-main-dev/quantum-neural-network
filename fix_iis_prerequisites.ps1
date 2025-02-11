# Fix missing Windows features for IIS
# Run this script as administrator
Write-Host "Fixing IIS Windows Features..." -ForegroundColor Cyan

# Required IIS Features
$features = @(
    "IIS-DefaultDocument",
    "IIS-DirectoryBrowsing",
    "IIS-HttpErrors",
    "IIS-StaticContent",
    "IIS-HttpLogging",
    "IIS-HttpTracing",
    "IIS-RequestMonitor",
    "IIS-HttpCompressionStatic",
    "IIS-HttpCompressionDynamic",
    "IIS-BasicAuthentication",
    "IIS-WindowsAuthentication",
    "IIS-NetFxExtensibility45",
    "IIS-ASPNET45",
    "IIS-ISAPIExtensions",
    "IIS-ISAPIFilter",
    "IIS-WebSockets",
    "IIS-ApplicationInit",
    "NetFx4Extended-ASPNET45",
    "IIS-ManagementConsole",
    "IIS-ManagementService"
)

# Enable each feature
foreach ($feature in $features) {
    Write-Host "Enabling feature: $feature"
    $result = dism /online /enable-feature /featurename:$feature /all
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully enabled $feature" -ForegroundColor Green
    } else {
        Write-Host "Failed to enable $feature" -ForegroundColor Red
        Write-Host $result
    }
}

# Install ASP.NET Core Module
Write-Host "`nInstalling ASP.NET Core Module..."
$tempDir = "$env:TEMP\dotnet_hosting"
New-Item -ItemType Directory -Force -Path $tempDir
$hostingBundleUrl = "https://download.visualstudio.microsoft.com/download/pr/7de08ae2-75e6-49b8-b04a-31526204fa7b/c1cee44a509495e4hidden_1ebed768/dotnet-hosting-6.0.0-win.exe"
$hostingBundlePath = "$tempDir\dotnet-hosting-bundle.exe"

try {
    Invoke-WebRequest -Uri $hostingBundleUrl -OutFile $hostingBundlePath
    Start-Process -FilePath $hostingBundlePath -ArgumentList '/install', '/quiet', '/norestart' -Wait
    Write-Host "ASP.NET Core Module installed successfully" -ForegroundColor Green
} catch {
    Write-Host "Failed to install ASP.NET Core Module: $_" -ForegroundColor Red
}

# Reset IIS to apply changes
Write-Host "`nResetting IIS..."
iisreset /restart

Write-Host "`nSetup complete. Please run validate_iis_setup.ps1 to verify the installation."