# Direct IIS configuration using appcmd.exe
Write-Host "Configuring IIS using appcmd.exe..." -ForegroundColor Cyan

$sites = @(
    @{
        Name = "PersonaManager"
        Port = 8083
        Path = "C:\inetpub\wwwroot\PersonaManager"
        Pool = "PersonaManagerPool"
    },
    @{
        Name = "QuantumTensor"
        Port = 8081
        Path = "C:\inetpub\wwwroot\QuantumTensor"
        Pool = "QuantumTensorPool"
    },
    @{
        Name = "MindsDBBridge"
        Port = 8082
        Path = "C:\inetpub\wwwroot\MindsDBBridge"
        Pool = "MindsDBBridgePool"
    },
    @{
        Name = "NeuralNetworkCore"
        Port = 8080
        Path = "C:\inetpub\wwwroot\NeuralNetworkCore"
        Pool = "NeuralNetworkCorePool"
    }
)

# Stop IIS
Write-Host "Stopping IIS..."
iisreset /stop

# Delete existing sites and app pools
foreach ($site in $sites) {
    Write-Host "Removing existing configuration for $($site.Name)..."
    & "$env:windir\system32\inetsrv\appcmd.exe" delete site "$($site.Name)" 2>$null
    & "$env:windir\system32\inetsrv\appcmd.exe" delete apppool "$($site.Pool)" 2>$null
}

# Create app pools and sites
foreach ($site in $sites) {
    Write-Host "`nConfiguring $($site.Name)..."
    
    # Create app pool
    Write-Host "Creating application pool $($site.Pool)..."
    & "$env:windir\system32\inetsrv\appcmd.exe" add apppool /name:$($site.Pool) /managedRuntimeVersion:v4.0 /managedPipelineMode:Integrated
    & "$env:windir\system32\inetsrv\appcmd.exe" set apppool "$($site.Pool)" /processModel.identityType:ApplicationPoolIdentity
    & "$env:windir\system32\inetsrv\appcmd.exe" set apppool "$($site.Pool)" /startMode:AlwaysRunning
    
    # Create site
    Write-Host "Creating website $($site.Name)..."
    & "$env:windir\system32\inetsrv\appcmd.exe" add site /name:$($site.Name) /bindings:"http://*:$($site.Port)" /physicalPath:$($site.Path)
    & "$env:windir\system32\inetsrv\appcmd.exe" set site "$($site.Name)" /applicationDefaults.applicationPool:$($site.Pool)
    
    # Create simple health check page
    if (-not (Test-Path $site.Path)) {
        New-Item -ItemType Directory -Path $site.Path -Force
    }
    
    $healthCheck = @"
<!DOCTYPE html>
<html>
<head><title>$($site.Name) Health Check</title></head>
<body>
    <h1>$($site.Name) is running</h1>
    <p>Port: $($site.Port)</p>
    <p>Time: $([DateTime]::Now)</p>
</body>
</html>
"@
    Set-Content -Path "$($site.Path)\health.html" -Value $healthCheck
}

# Start IIS
Write-Host "`nStarting IIS..."
iisreset /start

# Configure firewall rules
Write-Host "`nConfiguring firewall rules..."
foreach ($site in $sites) {
    $ruleName = "IIS $($site.Name) Port $($site.Port)"
    Remove-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $($site.Port) -Action Allow
}

Write-Host "`nConfiguration complete. Please verify the following:"
Write-Host "1. Test each site using: http://localhost:<port>/health.html"
Write-Host "2. Check application pools in IIS Manager"
Write-Host "3. Verify firewall rules are active"
Write-Host "4. Check Event Viewer for any errors"

# Display test URLs
Write-Host "`nTest URLs:"
foreach ($site in $sites) {
    Write-Host "$($site.Name): http://localhost:$($site.Port)/health.html"
}