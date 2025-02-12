Write-Host "Configuring sleep cycle service..." -ForegroundColor Cyan

# Import required modules
Import-Module WebAdministration

# Define sleep cycle app pool
$appPoolName = "DreamProcessor"
$siteName = "dream_processor"
$port = 8106

# Create and configure application pool
if (!(Test-Path IIS:\AppPools\$appPoolName)) {
    New-WebAppPool -Name $appPoolName
    Set-ItemProperty IIS:\AppPools\$appPoolName -name "managedRuntimeVersion" -value "v4.0"
    Set-ItemProperty IIS:\AppPools\$appPoolName -name "startMode" -value "AlwaysRunning"
    Set-ItemProperty IIS:\AppPools\$appPoolName -name "processModel.idleTimeout" -value "00:00:00"
}

# Configure FastCGI settings for Python
$pythonPath = ".venv\Scripts\python.exe"
$scriptProcessor = Get-ChildItem $pythonPath -ErrorAction SilentlyContinue
if ($scriptProcessor) {
    $fastCgiPath = "MACHINE/WEBROOT/APPHOST"
    Set-WebConfigurationProperty -PSPath $fastCgiPath -Filter "system.webServer/fastCgi" -Name "" -Value @{
        fullPath = $scriptProcessor.FullName
        arguments = "dream_processor.dream_state_manager.py"
        maxInstances = "1"
        instanceMaxRequests = "10000"
        idleTimeout = "0"
    }
}

# Create website if it doesn't exist
if (!(Get-Website -Name $siteName)) {
    New-Website -Name $siteName -Port $port -PhysicalPath "$PSScriptRoot\brain_capsules\dream_processor" -ApplicationPool $appPoolName
}

# Configure web.config for the dream processor
$webConfigContent = @"
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <handlers>
            <add name="PythonHandler" path="*" verb="*" modules="FastCgiModule" 
                 scriptProcessor="$($scriptProcessor.FullName)|$($scriptProcessor.FullName)"
                 resourceType="Unspecified" requireAccess="Script" />
        </handlers>
        <security>
            <requestFiltering>
                <requestLimits maxAllowedContentLength="30000000" />
            </requestFiltering>
        </security>
    </system.webServer>
</configuration>
"@

Set-Content -Path "$PSScriptRoot\brain_capsules\dream_processor\web.config" -Value $webConfigContent

# Update brain state configuration
$brainStateConfig = Get-Content "config\brain_state.json" | ConvertFrom-Json
$brainStateConfig.brain_state.core_components | Add-Member -NotePropertyName "dream_processor" -NotePropertyValue @{
    status = "configured"
    port = 8106
    service_type = "IIS"
    security_level = "maximum"
    local_only = $true
    sleep_cycle = @{
        enabled = $true
        auto_schedule = $true
        consolidation_threshold = 0.85
        cleanup_interval = "04:00:00"
    }
}

$brainStateConfig | ConvertTo-Json -Depth 10 | Set-Content "config\brain_state.json"

Write-Host "Dream processor service configured successfully" -ForegroundColor Green