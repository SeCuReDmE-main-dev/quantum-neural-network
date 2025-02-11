# Direct IIS Configuration Script using Microsoft.Web.Administration
Add-Type -Path "${env:windir}\system32\inetsrv\Microsoft.Web.Administration.dll"

Write-Host "Configuring IIS using native API..." -ForegroundColor Cyan

try {
    # Create Server Manager instance
    $manager = New-Object Microsoft.Web.Administration.ServerManager

    # Configure Application Pools
    $pools = @(
        @{
            Name = "PersonaManagerPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "QuantumTensorPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "MindsDBBridgePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "NeuralNetworkCorePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "BasalGangliaPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        }
    )

    foreach ($poolConfig in $pools) {
        $pool = $manager.ApplicationPools[$poolConfig.Name]
        if ($pool -eq $null) {
            $pool = $manager.ApplicationPools.Add($poolConfig.Name)
            $pool.ManagedRuntimeVersion = $poolConfig.RuntimeVersion
            $pool.ManagedPipelineMode = $poolConfig.PipelineMode
            Write-Host "Created application pool: $($poolConfig.Name)" -ForegroundColor Green
        } else {
            Write-Host "Application pool already exists: $($poolConfig.Name)" -ForegroundColor Yellow
        }
    }

    # Configure Sites
    $sites = @(
        @{
            Name = "PersonaManager"
            Port = 8083
            Pool = "PersonaManagerPool"
            Path = "C:\inetpub\wwwroot\PersonaManager"
        },
        @{
            Name = "QuantumTensor"
            Port = 8081
            Pool = "QuantumTensorPool"
            Path = "C:\inetpub\wwwroot\QuantumTensor"
        },
        @{
            Name = "MindsDBBridge"
            Port = 8082
            Pool = "MindsDBBridgePool"
            Path = "C:\inetpub\wwwroot\MindsDBBridge"
        },
        @{
            Name = "NeuralNetworkCore"
            Port = 8080
            Pool = "NeuralNetworkCorePool"
            Path = "C:\inetpub\wwwroot\NeuralNetworkCore"
        },
        @{
            Name = "BasalGanglia"
            Port = 8084
            Pool = "BasalGangliaPool"
            Path = "C:\inetpub\wwwroot\BasalGanglia"
        }
    )

    foreach ($siteConfig in $sites) {
        # Create directory if it doesn't exist
        if (-not (Test-Path $siteConfig.Path)) {
            New-Item -ItemType Directory -Path $siteConfig.Path -Force
        }

        $site = $manager.Sites[$siteConfig.Name]
        if ($site -eq $null) {
            $site = $manager.Sites.Add($siteConfig.Name, "http", "*:$($siteConfig.Port):", $siteConfig.Path)
            $site.Applications["/"].ApplicationPoolName = $siteConfig.Pool
            Write-Host "Created site: $($siteConfig.Name) on port $($siteConfig.Port)" -ForegroundColor Green
        } else {
            Write-Host "Site already exists: $($siteConfig.Name)" -ForegroundColor Yellow
        }
    }

    # Commit changes
    $manager.CommitChanges()
    Write-Host "Configuration committed successfully" -ForegroundColor Green

    # Configure Windows Firewall rules
    foreach ($siteConfig in $sites) {
        $ruleName = "IIS $($siteConfig.Name) Port $($siteConfig.Port)"
        if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName $ruleName `
                              -Direction Inbound `
                              -Protocol TCP `
                              -LocalPort $siteConfig.Port `
                              -Action Allow
            Write-Host "Created firewall rule: $ruleName" -ForegroundColor Green
        } else {
            Write-Host "Firewall rule already exists: $ruleName" -ForegroundColor Yellow
        }
    }

} catch {
    Write-Error "Error configuring IIS: $_"
    Write-Host "Stack Trace: $($_.Exception.StackTrace)" -ForegroundColor Red
} finally {
    if ($manager -ne $null) {
        $manager.Dispose()
    }
}

Write-Host "`nIIS Configuration Complete" -ForegroundColor Cyan
Write-Host "Please verify the following manually:"
Write-Host "1. Sites are running correctly (iisreset may be needed)"
Write-Host "2. Application pools are started"
Write-Host "3. Ports are accessible"
Write-Host "4. Firewall rules are active"