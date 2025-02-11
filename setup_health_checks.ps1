# Setup health check endpoints for brain parts
Write-Host "Setting up health check endpoints..." -ForegroundColor Cyan

$brainParts = @(
    # Level 1: Core Brain Parts
    @{ Name = "Cerebrum"; Port = 8090; IP = "10.0.0.163" },
    @{ Name = "Brainstem"; Port = 8091; IP = "10.0.0.164" },
    @{ Name = "Cerebellum"; Port = 8092; IP = "10.0.0.165" },
    # ... remaining parts with their IPs and ports
)

foreach ($part in $brainParts) {
    $sitePath = "C:\inetpub\wwwroot\$($part.Name)"
    $healthCheckPath = Join-Path $sitePath "health.html"
    
    if (-not (Test-Path $sitePath)) {
        New-Item -ItemType Directory -Path $sitePath -Force
    }
    
    $healthCheckContent = @"
<!DOCTYPE html>
<html>
<head>
    <title>$($part.Name) Health Check</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .status { color: green; }
        .details { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>$($part.Name) Status: <span class="status">Running</span></h1>
    <div class="details">
        <p>Port: $($part.Port)</p>
        <p>IP: $($part.IP)</p>
        <p>Time: $([DateTime]::Now)</p>
    </div>
</body>
</html>
"@
    
    Set-Content -Path $healthCheckPath -Value $healthCheckContent
    Write-Host "Created health check for $($part.Name) at $healthCheckPath" -ForegroundColor Green
}

Write-Host "`nHealth check endpoints created successfully!" -ForegroundColor Green
Write-Host "Access them at http://{ip}:{port}/health.html"