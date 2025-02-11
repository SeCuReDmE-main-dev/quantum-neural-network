# IIS Component verification script - does not require admin rights
Write-Host "Verifying IIS Components..." -ForegroundColor Cyan

# Function to check if a port is in use
function Test-Port {
    param($Port)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect('127.0.0.1', $Port)
        $tcp.Close()
        return $true
    } catch {
        return $false
    }
}

# Check if IIS service exists
$iisService = Get-Service -Name W3SVC -ErrorAction SilentlyContinue
Write-Host "IIS Service Status: $(if ($iisService) { $iisService.Status } else { 'Not Installed' })"

# Check if required directories exist
$directories = @(
    "C:\inetpub\wwwroot\PersonaManager",
    "C:\inetpub\wwwroot\QuantumTensor",
    "C:\inetpub\wwwroot\MindsDBBridge",
    "C:\inetpub\wwwroot\NeuralNetworkCore"
)

Write-Host "`nChecking directories..."
foreach ($dir in $directories) {
    Write-Host "$dir : $(if (Test-Path $dir) { 'Exists' } else { 'Missing' })"
}

# Check port availability
$ports = @(8080, 8081, 8082, 8083)
Write-Host "`nChecking ports..."
foreach ($port in $ports) {
    $inUse = Test-Port -Port $port
    Write-Host "Port $port : $(if ($inUse) { 'In Use' } else { 'Not responding' })"
}

# Check if Microsoft.Web.Administration.dll exists
$webAdminPath = "${env:windir}\system32\inetsrv\Microsoft.Web.Administration.dll"
Write-Host "`nChecking IIS components..."
Write-Host "Microsoft.Web.Administration.dll: $(if (Test-Path $webAdminPath) { 'Exists' } else { 'Missing' })"

# Determine next steps
Write-Host "`nStatus Summary:" -ForegroundColor Yellow
if (-not $iisService) {
    Write-Host "IIS needs to be installed using DISM commands" -ForegroundColor Red
    Write-Host "Run the following command as administrator:"
    Write-Host "dism /online /enable-feature /featurename:IIS-WebServerRole /featurename:IIS-WebServer /featurename:IIS-CommonHttpFeatures /featurename:IIS-ManagementConsole /all"
}

if (-not (Test-Path $webAdminPath)) {
    Write-Host "IIS Management components need to be installed" -ForegroundColor Red
    Write-Host "Run the following command as administrator:"
    Write-Host "dism /online /enable-feature /featurename:IIS-ManagementScriptingTools /featurename:IIS-IIS6ManagementCompatibility /all"
}

$missingDirs = $directories | Where-Object { -not (Test-Path $_) }
if ($missingDirs) {
    Write-Host "`nMissing directories need to be created:" -ForegroundColor Yellow
    $missingDirs | ForEach-Object { Write-Host $_ }
}

$unreachablePorts = $ports | Where-Object { -not (Test-Port $_) }
if ($unreachablePorts) {
    Write-Host "`nPorts not responding:" -ForegroundColor Yellow
    $unreachablePorts | ForEach-Object { Write-Host "Port $_" }
}