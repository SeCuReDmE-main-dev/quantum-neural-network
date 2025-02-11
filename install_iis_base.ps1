# This script must be run as administrator
#Requires -RunAsAdministrator

Write-Host "Installing IIS base components..."

Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServerRole
Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServer
Enable-WindowsOptionalFeature -Online -FeatureName IIS-CommonHttpFeatures
Enable-WindowsOptionalFeature -Online -FeatureName IIS-StaticContent
Enable-WindowsOptionalFeature -Online -FeatureName IIS-DefaultDocument
Enable-WindowsOptionalFeature -Online -FeatureName IIS-DirectoryBrowsing
Enable-WindowsOptionalFeature -Online -FeatureName IIS-HttpErrors
Enable-WindowsOptionalFeature -Online -FeatureName IIS-ApplicationDevelopment

# Open required ports
Write-Host "Opening required ports in Windows Firewall..."

$ports = @(8080, 8081, 8082, 8083)
foreach ($port in $ports) {
    $ruleName = "TensorZero IIS Port $port"
    New-NetFirewallRule -DisplayName $ruleName `
                       -Direction Inbound `
                       -Protocol TCP `
                       -LocalPort $port `
                       -Action Allow `
                       -ErrorAction SilentlyContinue
    
    Write-Host "Opened port $port"
}

Write-Host "Initial IIS setup complete. Please run setup_iis_apache.ps1 next to configure the services."