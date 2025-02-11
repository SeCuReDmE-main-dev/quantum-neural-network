# This script must be run as administrator
#Requires -RunAsAdministrator

Write-Host "Installing IIS components..."

# Enable IIS features using DISM
$features = @(
    "IIS-WebServerRole",
    "IIS-WebServer",
    "IIS-CommonHttpFeatures",
    "IIS-ManagementConsole",
    "IIS-ManagementScriptingTools",
    "IIS-ASPNET45",
    "IIS-ASPNET",
    "IIS-NetFxExtensibility45",
    "IIS-CGI",
    "IIS-ISAPIExtensions",
    "IIS-ISAPIFilter",
    "IIS-WebSockets"
)

foreach ($feature in $features) {
    Write-Host "Enabling feature: $feature"
    DISM /Online /Enable-Feature /FeatureName:$feature /All
}

# Install IIS Remote Management
Write-Host "Installing IIS Remote Management..."
DISM /Online /Enable-Feature /FeatureName:IIS-WebServerManagementTools /FeatureName:IIS-ManagementService /All

# Configure Web Management Service (WMSVC) for remote management
Write-Host "Configuring Web Management Service..."
Set-ItemProperty -Path HKLM:\SOFTWARE\Microsoft\WebManagement\Server -Name EnableRemoteManagement -Value 1
Start-Service WMSVC
Set-Service -Name WMSVC -StartupType Automatic

# Configure firewall rules
Write-Host "Configuring firewall rules..."
$ports = @(8080, 8081, 8082, 8083)
foreach ($port in $ports) {
    $ruleName = "IIS Port $port"
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $port -Action Allow
}

# Enable remote management ports
New-NetFirewallRule -DisplayName "IIS Remote Management" -Direction Inbound -Protocol TCP -LocalPort 8172 -Action Allow

Write-Host "IIS installation complete. Please run setup_iis_apache.ps1 next to configure the services."