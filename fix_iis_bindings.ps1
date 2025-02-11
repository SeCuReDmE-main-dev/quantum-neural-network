# Fix IIS port bindings and app pools
Add-Type -AssemblyName System.DirectoryServices

Write-Host "Fixing IIS port bindings and application pools..." -ForegroundColor Cyan

try {
    # Get IIS Admin COM object
    $iis = [ADSI]"IIS://localhost/W3SVC"
    
    # Site configurations
    $siteConfigs = @(
        @{
            Name = "PersonaManager"
            Port = 8083
            Path = "C:\inetpub\wwwroot\PersonaManager"
            AppPool = "PersonaManagerPool"
        },
        @{
            Name = "QuantumTensor"
            Port = 8081
            Path = "C:\inetpub\wwwroot\QuantumTensor"
            AppPool = "QuantumTensorPool"
        },
        @{
            Name = "MindsDBBridge"
            Port = 8082
            Path = "C:\inetpub\wwwroot\MindsDBBridge"
            AppPool = "MindsDBBridgePool"
        },
        @{
            Name = "NeuralNetworkCore"
            Port = 8080
            Path = "C:\inetpub\wwwroot\NeuralNetworkCore"
            AppPool = "NeuralNetworkCorePool"
        }
    )

    foreach ($config in $siteConfigs) {
        Write-Host "`nConfiguring site: $($config.Name)" -ForegroundColor Yellow
        
        # Create or update binding info
        $bindingInfo = ":$($config.Port):$($config.Path)"
        
        # Check if site exists
        $site = Get-ChildItem IIS:\Sites | Where-Object { $_.Name -eq $config.Name }
        
        if (-not $site) {
            Write-Host "Creating new site: $($config.Name)"
            
            # Create application pool if it doesn't exist
            $appPool = [ADSI]"IIS://localhost/W3SVC/AppPools/$($config.AppPool)"
            if (-not $appPool.Name) {
                Write-Host "Creating application pool: $($config.AppPool)"
                $newPool = $iis.Create("IIsApplicationPool", $config.AppPool)
                $newPool.SetInfo()
            }
            
            # Create the site
            $newSite = $iis.Create("IIsWebServer", $config.Name)
            $newSite.Put("ServerBindings", $bindingInfo)
            $newSite.Put("ServerComment", $config.Name)
            $newSite.Put("Path", $config.Path)
            $newSite.Put("AppPoolId", $config.AppPool)
            $newSite.SetInfo()
            
            # Start the site
            $newSite.Start()
        } else {
            Write-Host "Updating existing site: $($config.Name)"
            $site.ServerBindings = $bindingInfo
            $site.ApplicationPool = $config.AppPool
            $site.PhysicalPath = $config.Path
            $site.Update()
        }
        
        # Verify site is running
        $siteStatus = Get-Website -Name $config.Name -ErrorAction SilentlyContinue
        Write-Host "Site status: $($siteStatus.State)"
    }
    
    # Restart IIS to apply changes
    Write-Host "`nRestarting IIS..." -ForegroundColor Yellow
    iisreset /restart
    
    Write-Host "`nVerifying port bindings..." -ForegroundColor Yellow
    foreach ($config in $siteConfigs) {
        $connection = Test-NetConnection -ComputerName localhost -Port $config.Port -WarningAction SilentlyContinue
        Write-Host "Port $($config.Port) ($($config.Name)): $(if ($connection.TcpTestSucceeded) { 'Responding' } else { 'Not responding' })"
    }

} catch {
    Write-Error "Error configuring IIS: $_"
    Write-Host $_.Exception.Message
    Write-Host $_.Exception.StackTrace
} finally {
    Write-Host "`nConfiguration complete. If ports are still not responding, please check:"
    Write-Host "1. Windows Firewall settings"
    Write-Host "2. Application pool identity permissions"
    Write-Host "3. IIS site bindings in IIS Manager"
    Write-Host "4. Event Viewer for specific errors"
}

# Add health check files to each site
foreach ($config in $siteConfigs) {
    $healthCheck = @"
<!DOCTYPE html>
<html>
<head><title>$($config.Name) Health Check</title></head>
<body>
    <h1>$($config.Name) is running</h1>
    <p>Port: $($config.Port)</p>
    <p>Time: $([DateTime]::Now)</p>
</body>
</html>
"@
    
    Set-Content -Path "$($config.Path)\health.html" -Value $healthCheck
}