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

# Configure FastCGI Application
$fastCgiPath = "IIS:\FastCgi\-"
$pythonPath = "C:\Python310\python.exe"
$wfastcgiPath = "C:\Python310\Scripts\wfastcgi.py"

# Create FastCGI application
Add-WebConfiguration /system.webServer/fastCgi -Value @{
    fullPath = $pythonPath
    arguments = $wfastcgiPath
    maxInstances = 32
    idleTimeout = 300
    activityTimeout = 300
    requestTimeout = 900
    instanceMaxRequests = 10000
    protocol = "NamedPipe"
    flushNamedPipe = "False"
}

# Configure environment variables
$envVars = @{
    "PYTHONPATH" = "C:\Users\jeans\OneDrive\Desktop\letdo it\AdvancedMaterialScience"
    "WSGI_HANDLER" = "quantum_neural.neural_network.persona_manager.app"
}

foreach ($key in $envVars.Keys) {
    Set-WebConfigurationProperty -Filter "system.webServer/fastCgi/application[@fullPath='$pythonPath']/environmentVariables/environmentVariable[@name='$key']" `
        -Name "value" `
        -Value $envVars[$key]
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

Write-Host "Configuration complete. Please verify the following:" -ForegroundColor Green
Write-Host "1. IIS Service is running"
Write-Host "2. FastCGI handler is configured"
Write-Host "3. Application pools are created"
Write-Host "4. URL rewrite rules are set up (if module is installed)"
Write-Host "`nNext steps:"
Write-Host "1. Complete the Web Platform Installer installation"
Write-Host "2. Install the .NET Core Hosting Bundle"
Write-Host "3. Restart IIS using 'iisreset' command"