# Verify IIS installation and configure FastCGI
Write-Host "Verifying IIS installation and configuring Python FastCGI..." -ForegroundColor Cyan

# Check IIS installation
$iisInstalled = Get-Service W3SVC -ErrorAction SilentlyContinue
if (-not $iisInstalled) {
    Write-Host "IIS is not installed. Please complete the DISM installation first." -ForegroundColor Red
    exit 1
}

# Install WFastCGI if not present
if (-not (Test-Path "C:\Python310\Scripts\wfastcgi.py")) {
    Write-Host "Installing WFastCGI..." -ForegroundColor Yellow
    pip install wfastcgi
    wfastcgi-enable
}

# Requires -RunAsAdministrator
Write-Host "Configuring FastCGI for Python..." -ForegroundColor Cyan

$pythonPath = "C:\Users\jeans\OneDrive\Desktop\letdo it\AdvancedMaterialScience\venv\Scripts\python.exe"
$wfastcgiPath = "C:\Users\jeans\OneDrive\Desktop\letdo it\AdvancedMaterialScience\venv\Scripts\wfastcgi.py"

# Configure FastCGI Settings
$fastCgiSection = "system.webServer/fastCgi"
$fastCgiArgs = @{
    fullPath = $pythonPath
    arguments = $wfastcgiPath
    maxInstances = "4"
    idleTimeout = "1800"
}

# Add FastCGI application
try {
    Import-Module WebAdministration
    
    # Remove existing configuration if any
    Clear-WebConfiguration -Filter "$fastCgiSection/application[@fullPath='$pythonPath']"
    
    # Add new configuration
    Add-WebConfiguration -Filter $fastCgiSection -Value $fastCgiArgs -PSPath "MACHINE/WEBROOT/APPHOST"
    
    # Set environment variables
    $envVars = @{
        "PYTHONPATH" = "C:\Users\jeans\OneDrive\Desktop\letdo it\AdvancedMaterialScience"
        "WSGI_HANDLER" = "app.app"
    }
    
    foreach ($var in $envVars.GetEnumerator()) {
        Set-WebConfigurationProperty -Filter "$fastCgiSection/application[@fullPath='$pythonPath']/environmentVariables" `
            -Name "." `
            -Value @{name=$var.Key; value=$var.Value}
    }
    
    Write-Host "FastCGI configuration completed successfully" -ForegroundColor Green
    Write-Host "Please restart IIS using 'iisreset' command" -ForegroundColor Yellow
} catch {
    Write-Error "Failed to configure FastCGI: $_"
    exit 1
}

# Verify app pools
$appPools = @(
    "PersonaManagerPool",
    "QuantumTensorPool",
    "NeuralNetworkCorePool",
    "MindsDBBridgePool"
)

foreach ($pool in $appPools) {
    if (-not (Test-Path "IIS:\AppPools\$pool")) {
        Write-Host "Creating application pool: $pool" -ForegroundColor Yellow
        New-WebAppPool -Name $pool -Force
        Set-ItemProperty "IIS:\AppPools\$pool" -Name "managedRuntimeVersion" -Value ""
        Set-ItemProperty "IIS:\AppPools\$pool" -Name "startMode" -Value "AlwaysRunning"
        Set-ItemProperty "IIS:\AppPools\$pool" -Name "processModel.idleTimeout" -Value ([TimeSpan]::FromMinutes(0))
    }
}

# Create URL rewrite rules if module is installed
$rewriteModule = Get-WebGlobalModule -Name "RewriteModule"
if ($rewriteModule) {
    Write-Host "Configuring URL rewrite rules..." -ForegroundColor Yellow
    
    $rewriteRules = @{
        "PersonaAPI" = @{
            pattern = "^api/persona/(.*)"
            url = "http://localhost:8083/{R:1}"
        }
        "QuantumAPI" = @{
            pattern = "^api/quantum/(.*)"
            url = "http://localhost:8081/{R:1}"
        }
        "NeuralAPI" = @{
            pattern = "^api/neural/(.*)"
            url = "http://localhost:8080/{R:1}"
        }
        "MindsDBAPI" = @{
            pattern = "^api/mindsdb/(.*)"
            url = "http://localhost:8082/{R:1}"
        }
    }

    foreach ($rule in $rewriteRules.GetEnumerator()) {
        Add-WebConfigurationProperty -PSPath 'MACHINE/WEBROOT/APPHOST' -Filter "system.webServer/rewrite/rules" -Name "." -Value @{
            name = $rule.Key
            patternSyntax = "ECMAScript"
            stopProcessing = "True"
        }
        Set-WebConfigurationProperty -PSPath 'MACHINE/WEBROOT/APPHOST' -Filter "system.webServer/rewrite/rules/rule[@name='$($rule.Key)']/match" -Name "url" -Value $rule.Value.pattern
        Set-WebConfigurationProperty -PSPath 'MACHINE/WEBROOT/APPHOST' -Filter "system.webServer/rewrite/rules/rule[@name='$($rule.Key)']/action" -Name "type" -Value "Rewrite"
        Set-WebConfigurationProperty -PSPath 'MACHINE/WEBROOT/APPHOST' -Filter "system.webServer/rewrite/rules/rule[@name='$($rule.Key)']/action" -Name "url" -Value $rule.Value.url
    }
}

# Configure FastCGI settings for brain parts
Write-Host "Configuring FastCGI settings..." -ForegroundColor Cyan

# Import required module
Import-Module WebAdministration

$brainParts = @(
    # Level 1: Core Processing
    @{
        Name = "Cerebrum"
        IP = "10.0.0.163" 
        Port = 8090
        ProcessModel = @{
            MaxInstances = 4
            IdleTimeout = 1800
        }
    },
    @{
        Name = "Brainstem"
        IP = "10.0.0.164"
        Port = 8091
        ProcessModel = @{
            MaxInstances = 2
            IdleTimeout = 1800
        }
    },
    @{
        Name = "Cerebellum"
        IP = "10.0.0.165"
        Port = 8092
        ProcessModel = @{
            MaxInstances = 4
            IdleTimeout = 1800
        }
    }
    # Add other brain parts here...
)

foreach ($part in $brainParts) {
    $webConfigPath = "C:\inetpub\wwwroot\$($part.Name)\web.config"
    
    $webConfigContent = @"
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <handlers>
            <add name="PythonHandler" path="*" verb="*" modules="FastCgiModule" 
                 scriptProcessor="C:\Python\python.exe|C:\Python\Lib\site-packages\wfastcgi.py" 
                 resourceType="Unspecified" />
        </handlers>
        <fastCgi>
            <application fullPath="C:\Python\python.exe" 
                        arguments="C:\Python\Lib\site-packages\wfastcgi.py"
                        maxInstances="$($part.ProcessModel.MaxInstances)"
                        idleTimeout="$($part.ProcessModel.IdleTimeout)">
                <environmentVariables>
                    <environmentVariable name="PYTHONPATH" value="C:\inetpub\wwwroot\$($part.Name)" />
                    <environmentVariable name="WSGI_HANDLER" value="app.app" />
                    <environmentVariable name="BRAIN_PART" value="$($part.Name)" />
                    <environmentVariable name="QUANTUM_BRIDGE_ENABLED" value="true" />
                    <environmentVariable name="PHI_FRAMEWORK_ENABLED" value="true" />
                    <environmentVariable name="PORT" value="$($part.Port)" />
                    <environmentVariable name="IP" value="$($part.IP)" />
                </environmentVariables>
            </application>
        </fastCgi>
        <security>
            <requestFiltering>
                <requestLimits maxAllowedContentLength="30000000" />
            </requestFiltering>
        </security>
    </system.webServer>
</configuration>
"@

    Set-Content -Path $webConfigPath -Value $webConfigContent
    Write-Host "Configured FastCGI for $($part.Name)" -ForegroundColor Green
}

Write-Host "`nFastCGI configuration complete!" -ForegroundColor Green

Write-Host "Configuration complete. Please verify the following:" -ForegroundColor Green
Write-Host "1. IIS Service is running"
Write-Host "2. FastCGI handler is configured"
Write-Host "3. Application pools are created"
Write-Host "4. URL rewrite rules are set up (if module is installed)"
Write-Host "`nNext steps:"
Write-Host "1. Complete the Web Platform Installer installation"
Write-Host "2. Install the .NET Core Hosting Bundle"
Write-Host "3. Restart IIS using 'iisreset' command"