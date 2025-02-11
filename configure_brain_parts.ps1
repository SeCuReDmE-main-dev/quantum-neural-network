# Requires -RunAsAdministrator

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "This script requires administrator privileges. Please run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

# Direct IIS Configuration Script for Brain Parts
Add-Type -Path "${env:windir}\system32\inetsrv\Microsoft.Web.Administration.dll"

Write-Host "Configuring Brain Parts in IIS..." -ForegroundColor Cyan

$brainParts = @(
    # Level 1: Core Brain Parts
    @{ Name = "Cerebrum"; Port = 8090; Pool = "CerebrumPool"; IP = "10.0.0.163" },
    @{ Name = "Brainstem"; Port = 8091; Pool = "BrainstemPool"; IP = "10.0.0.164" },
    @{ Name = "Cerebellum"; Port = 8092; Pool = "CerebellumPool"; IP = "10.0.0.165" },

    # Level 2: Hemispheres
    @{ Name = "RightHemisphere"; Port = 8093; Pool = "RightHemispherePool"; IP = "10.0.0.166" },
    @{ Name = "LeftHemisphere"; Port = 8094; Pool = "LeftHemispherePool"; IP = "10.0.0.167" },
    @{ Name = "CorpusCallosum"; Port = 8095; Pool = "CorpusCallosumPool"; IP = "10.0.0.168" },

    # Level 3: Lobes
    @{ Name = "OccipitalLobe"; Port = 8096; Pool = "OccipitalLobePool"; IP = "10.0.0.169" },
    @{ Name = "ParietalLobe"; Port = 8097; Pool = "ParietalLobePool"; IP = "10.0.0.170" },
    @{ Name = "TemporalLobe"; Port = 8098; Pool = "TemporalLobePool"; IP = "10.0.0.171" },
    @{ Name = "FrontalLobe"; Port = 8099; Pool = "FrontalLobePool"; IP = "10.0.0.172" },

    # Level 4: CSN and PSN Communication
    @{ Name = "Fossae"; Port = 8100; Pool = "FossaePool"; IP = "10.0.0.173" },

    # Level 5: Cortical Features
    @{ Name = "Gyrus"; Port = 8101; Pool = "GyrusPool"; IP = "10.0.0.174" },
    @{ Name = "Sulcus"; Port = 8102; Pool = "SulcusPool"; IP = "10.0.0.175" },

    # Deep Structures
    @{ Name = "Thalamus"; Port = 8103; Pool = "ThalamusPool"; IP = "10.0.0.176" },
    @{ Name = "Hypothalamus"; Port = 8104; Pool = "HypothalamusPool"; IP = "10.0.0.177" },
    @{ Name = "PituitaryGland"; Port = 8105; Pool = "PituitaryGlandPool"; IP = "10.0.0.178" },
    @{ Name = "PinealGland"; Port = 8106; Pool = "PinealGlandPool"; IP = "10.0.0.179" },
    @{ Name = "LimbicSystem"; Port = 8107; Pool = "LimbicSystemPool"; IP = "10.0.0.180" },
    @{ Name = "BasalGanglia"; Port = 8108; Pool = "BasalGangliaPool"; IP = "10.0.0.181" },

    # Memory and Personality
    @{ Name = "Hippocampus"; Port = 8109; Pool = "HippocampusPool"; IP = "10.0.0.182" },
    @{ Name = "PrefrontalCortex"; Port = 8110; Pool = "PrefrontalCortexPool"; IP = "10.0.0.183" },

    # Neural Communication
    @{ Name = "CranialNerves"; Port = 8111; Pool = "CranialNervesPool"; IP = "10.0.0.184" },

    # Protective Layers
    @{ Name = "DuraMater"; Port = 8112; Pool = "DuraMaterPool"; IP = "10.0.0.185" },
    @{ Name = "ArachnoidMater"; Port = 8113; Pool = "ArachnoidMaterPool"; IP = "10.0.0.186" },
    @{ Name = "PiaMater"; Port = 8114; Pool = "PiaMaterPool"; IP = "10.0.0.187" },

    # Wave Patterns
    @{ Name = "WavePattern"; Port = 8115; Pool = "WavePatternPool"; IP = "10.0.0.188" }
)

try {
    $manager = New-Object Microsoft.Web.Administration.ServerManager

    foreach ($part in $brainParts) {
        # Create Application Pool
        $pool = $manager.ApplicationPools[$part.Pool]
        if ($pool -eq $null) {
            $pool = $manager.ApplicationPools.Add($part.Pool)
            $pool.ManagedRuntimeVersion = "v4.0"
            $pool.ManagedPipelineMode = "Integrated"
            Write-Host "Created application pool: $($part.Pool)" -ForegroundColor Green
        }

        # Create Website Directory
        $sitePath = "C:\inetpub\wwwroot\$($part.Name)"
        if (-not (Test-Path $sitePath)) {
            New-Item -ItemType Directory -Path $sitePath -Force
            Write-Host "Created directory: $sitePath" -ForegroundColor Green
        }

        # Create Website
        $site = $manager.Sites[$part.Name]
        if ($site -eq $null) {
            $binding = "*:$($part.Port):$($part.IP)"
            $site = $manager.Sites.Add($part.Name, "http", $binding, $sitePath)
            $site.Applications["/"].ApplicationPoolName = $part.Pool
            Write-Host "Created website: $($part.Name) on port $($part.Port)" -ForegroundColor Green
        }

        # Create web.config
        $webConfigPath = Join-Path $sitePath "web.config"
        if (-not (Test-Path $webConfigPath)) {
            $webConfigContent = @"
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <handlers>
            <add name="PythonHandler" path="*" verb="*" modules="FastCgiModule" 
                 scriptProcessor="C:\Python\python.exe|C:\Python\Lib\site-packages\wfastcgi.py" 
                 resourceType="Unspecified" />
        </handlers>
        <security>
            <requestFiltering>
                <requestLimits maxAllowedContentLength="30000000" />
            </requestFiltering>
        </security>
        <fastCgi>
            <application fullPath="C:\Python\python.exe" 
                        arguments="C:\Python\Lib\site-packages\wfastcgi.py"
                        maxInstances="4"
                        idleTimeout="1800">
                <environmentVariables>
                    <environmentVariable name="PYTHONPATH" value="C:\inetpub\wwwroot\$($part.Name)" />
                    <environmentVariable name="WSGI_HANDLER" value="app.app" />
                    <environmentVariable name="BRAIN_PART" value="$($part.Name)" />
                    <environmentVariable name="PORT" value="$($part.Port)" />
                    <environmentVariable name="IP" value="$($part.IP)" />
                </environmentVariables>
            </application>
        </fastCgi>
    </system.webServer>
</configuration>
"@
            Set-Content -Path $webConfigPath -Value $webConfigContent
            Write-Host "Created web.config for $($part.Name)" -ForegroundColor Green
        }

        # Configure firewall rule
        $ruleName = "IIS $($part.Name) Port $($part.Port)"
        if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName $ruleName `
                              -Direction Inbound `
                              -Protocol TCP `
                              -LocalPort $part.Port `
                              -Action Allow
            Write-Host "Created firewall rule: $ruleName" -ForegroundColor Green
        }
    }

    $manager.CommitChanges()
    Write-Host "`nConfiguration complete!" -ForegroundColor Green

} catch {
    Write-Error "Error configuring IIS: $_"
    Write-Host $_.Exception.StackTrace -ForegroundColor Red
} finally {
    if ($manager -ne $null) {
        $manager.Dispose()
    }
}

# Restart IIS to apply changes
Write-Host "`nRestarting IIS..." -ForegroundColor Cyan
iisreset /restart