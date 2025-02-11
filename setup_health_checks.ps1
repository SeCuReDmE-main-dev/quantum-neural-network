# Requires -RunAsAdministrator

Write-Host "Setting up health check endpoints..." -ForegroundColor Cyan

$brainParts = @(
    # Level 1: Core Brain Parts
    @{ Name = "Cerebrum"; Port = 8090; IP = "10.0.0.163" };
    @{ Name = "Brainstem"; Port = 8091; IP = "10.0.0.164" };
    @{ Name = "Cerebellum"; Port = 8092; IP = "10.0.0.165" };
    # Level 2: Hemispheres
    @{ Name = "RightHemisphere"; Port = 8093; IP = "10.0.0.166" };
    @{ Name = "LeftHemisphere"; Port = 8094; IP = "10.0.0.167" };
    @{ Name = "CorpusCallosum"; Port = 8095; IP = "10.0.0.168" };
    # Level 3: Lobes
    @{ Name = "OccipitalLobe"; Port = 8096; IP = "10.0.0.169" };
    @{ Name = "ParietalLobe"; Port = 8097; IP = "10.0.0.170" };
    @{ Name = "TemporalLobe"; Port = 8098; IP = "10.0.0.171" };
    @{ Name = "FrontalLobe"; Port = 8099; IP = "10.0.0.172" };
    # Level 4: CSN and PSN Communication
    @{ Name = "Fossae"; Port = 8100; IP = "10.0.0.173" };
    # Level 5: Cortical Features
    @{ Name = "Gyrus"; Port = 8101; IP = "10.0.0.174" };
    @{ Name = "Sulcus"; Port = 8102; IP = "10.0.0.175" };
    # Deep Structures
    @{ Name = "Thalamus"; Port = 8103; IP = "10.0.0.176" };
    @{ Name = "Hypothalamus"; Port = 8104; IP = "10.0.0.177" };
    @{ Name = "PituitaryGland"; Port = 8105; IP = "10.0.0.178" };
    @{ Name = "PinealGland"; Port = 8106; IP = "10.0.0.179" };
    @{ Name = "LimbicSystem"; Port = 8107; IP = "10.0.0.180" };
    @{ Name = "BasalGanglia"; Port = 8108; IP = "10.0.0.181" };
    @{ Name = "Hippocampus"; Port = 8109; IP = "10.0.0.182" };
    @{ Name = "PrefrontalCortex"; Port = 8110; IP = "10.0.0.183" };
    @{ Name = "CranialNerves"; Port = 8111; IP = "10.0.0.184" };
    @{ Name = "DuraMater"; Port = 8112; IP = "10.0.0.185" };
    @{ Name = "ArachnoidMater"; Port = 8113; IP = "10.0.0.186" };
    @{ Name = "PiaMater"; Port = 8114; IP = "10.0.0.187" };
    @{ Name = "WavePattern"; Port = 8115; IP = "10.0.0.188" }
)

# Create the main health checks directory if it doesn't exist
$healthChecksRoot = "C:\inetpub\wwwroot\HealthChecks"
if (-not (Test-Path $healthChecksRoot)) {
    New-Item -ItemType Directory -Path $healthChecksRoot -Force
}

foreach ($part in $brainParts) {
    $sitePath = "C:\inetpub\wwwroot\$($part.Name)"
    $healthCheckPath = Join-Path $sitePath "health.html"
    
    # Create site directory if it doesn't exist
    if (-not (Test-Path $sitePath)) {
        New-Item -ItemType Directory -Path $sitePath -Force
    }
    
    $healthCheckContent = @"
<!DOCTYPE html>
<html>
<head>
    <title>$($part.Name) Health Check</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { color: green; font-weight: bold; }
        .details { margin-top: 20px; }
        .info { color: #666; }
        .timestamp { font-size: 0.9em; color: #999; }
    </style>
</head>
<body>
    <div class="container">
        <h1>$($part.Name) Status: <span class="status">Running</span></h1>
        <div class="details">
            <p class="info"><strong>Port:</strong> $($part.Port)</p>
            <p class="info"><strong>IP:</strong> $($part.IP)</p>
            <p class="info"><strong>Brain Layer:</strong> $($part.Name -replace '([A-Z])', ' $1').Trim()</p>
            <p class="timestamp">Last Updated: $([DateTime]::Now.ToString("yyyy-MM-dd HH:mm:ss"))</p>
        </div>
    </div>
</body>
</html>
"@
    
    Set-Content -Path $healthCheckPath -Value $healthCheckContent
    Write-Host "Created health check for $($part.Name) at $healthCheckPath" -ForegroundColor Green
    
    # Create a symlink in the HealthChecks directory
    $healthCheckLink = Join-Path $healthChecksRoot "$($part.Name).html"
    if (-not (Test-Path $healthCheckLink)) {
        New-Item -ItemType SymbolicLink -Path $healthCheckLink -Target $healthCheckPath -Force
    }
}

Write-Host "`nHealth check endpoints created successfully!" -ForegroundColor Green
Write-Host "Access individual health checks at:" -ForegroundColor Cyan
Write-Host "http://{ip}:{port}/health.html"
Write-Host "`nOr view all health checks at:" -ForegroundColor Cyan
Write-Host "http://localhost/HealthChecks/"

# Create index page for all health checks
$indexContent = @"
<!DOCTYPE html>
<html>
<head>
    <title>Brain Parts Health Status</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .health-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
        .health-item { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; }
        .health-item h3 { margin-top: 0; color: #0066cc; }
        .timestamp { font-size: 0.8em; color: #666; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Brain Parts Health Status</h1>
        <div class="health-grid">
$(foreach ($part in $brainParts) {
@"
            <div class="health-item">
                <h3>$($part.Name)</h3>
                <p>IP: $($part.IP):$($part.Port)</p>
                <a href="http://$($part.IP):$($part.Port)/health.html" target="_blank">View Details</a>
            </div>
"@
})
        </div>
        <p class="timestamp">Last Updated: $([DateTime]::Now.ToString("yyyy-MM-dd HH:mm:ss"))</p>
    </div>
</body>
</html>
"@

Set-Content -Path (Join-Path $healthChecksRoot "index.html") -Value $indexContent
Write-Host "`nCreated health checks index page" -ForegroundColor Green