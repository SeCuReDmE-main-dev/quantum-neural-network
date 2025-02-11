# Elevate to admin if not already
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {    
    $arguments = "& '" + $myinvocation.mycommand.definition + "'"
    Start-Process powershell -Verb runAs -ArgumentList $arguments
    Break
}

# Install WebAdministration Module if not present
Write-Output "Installing and importing WebAdministration module..."
Install-WindowsFeature -Name Web-Scripting-Tools
Import-Module WebAdministration -ErrorAction SilentlyContinue
if (-not (Get-Module -Name WebAdministration)) {
    Write-Output "Installing WebAdministration module..."
    Install-Module -Name WebAdministration -Force -AllowClobber
    Import-Module WebAdministration
}

# Create IIS drive if it doesn't exist
if (-not (Get-PSDrive -Name IIS -ErrorAction SilentlyContinue)) {
    New-PSDrive -Name IIS -PSProvider Registry -Root HKLM:\SOFTWARE\Microsoft\InetStp
}

# Step 1: Install IIS and Required Features
Write-Output "Installing IIS and required features..."
$features = @(
    "Web-Server",
    "Web-WebServer",
    "Web-Common-Http",
    "Web-Default-Doc",
    "Web-Dir-Browsing",
    "Web-Http-Errors",
    "Web-Static-Content",
    "Web-Http-Redirect",
    "Web-Health",
    "Web-Http-Logging",
    "Web-Custom-Logging",
    "Web-Log-Libraries",
    "Web-Request-Monitor",
    "Web-Http-Tracing",
    "Web-Performance",
    "Web-Stat-Compression",
    "Web-Dyn-Compression",
    "Web-Security",
    "Web-Filtering",
    "Web-Basic-Auth",
    "Web-Windows-Auth",
    "Web-App-Dev",
    "Web-Net-Ext45",
    "Web-AppInit",
    "Web-Mgmt-Tools",
    "Web-Scripting-Tools"
)

foreach ($feature in $features) {
    try {
        Install-WindowsFeature -Name $feature -ErrorAction Stop
        Write-Output "Installed feature: $feature"
    } catch {
        Write-Warning "Failed to install feature: $feature"
        Write-Warning $_.Exception.Message
    }
}

# Step 2: Configure IIS Ports and Bindings
Write-Output "Configuring IIS ports..."

# Define ports for different components
$ports = @{
    "NeuralNetworkCore" = 8080
    "QuantumTensor" = 8081
    "MindsDBBridge" = 8082
    "PersonaManager" = 8083
}

# Create and configure sites
foreach ($site in $ports.GetEnumerator()) {
    $siteName = $site.Key
    $port = $site.Value
    $sitePath = "C:\inetpub\wwwroot\$siteName"
    
    # Create site directory if it doesn't exist
    if (-not (Test-Path $sitePath)) {
        New-Item -ItemType Directory -Path $sitePath -Force
    }
    
    # Remove existing site if it exists
    if (Get-Website -Name $siteName -ErrorAction SilentlyContinue) {
        Remove-Website -Name $siteName
    }

    # Create new website with proper binding
    New-Website -Name $siteName -PhysicalPath $sitePath -Port $port -Force

    # Create web.config for each site
    $webConfig = @"
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <handlers>
            <remove name="WebDAV" />
            <remove name="ExtensionlessUrlHandler-Integrated-4.0" />
            <add name="ExtensionlessUrlHandler-Integrated-4.0" path="*." verb="*" type="System.Web.Handlers.TransferRequestHandler" preCondition="integratedMode,runtimeVersionv4.0" />
        </handlers>
        <security>
            <requestFiltering>
                <requestLimits maxAllowedContentLength="30000000" />
            </requestFiltering>
        </security>
        <httpProtocol>
            <customHeaders>
                <add name="Access-Control-Allow-Origin" value="*" />
                <add name="Access-Control-Allow-Headers" value="Content-Type" />
                <add name="Access-Control-Allow-Methods" value="GET, POST, PUT, DELETE, OPTIONS" />
            </customHeaders>
        </httpProtocol>
    </system.webServer>
</configuration>
"@
    Set-Content -Path "$sitePath\web.config" -Value $webConfig

    # Configure application pool
    $poolName = "$siteName" + "Pool"
    if (Test-Path "IIS:\AppPools\$poolName") {
        Remove-WebAppPool -Name $poolName
    }

    $pool = New-WebAppPool -Name $poolName
    Set-ItemProperty -Path "IIS:\AppPools\$poolName" -Name "managedRuntimeVersion" -Value "v4.0"
    Set-ItemProperty -Path "IIS:\AppPools\$poolName" -Name "startMode" -Value "AlwaysRunning"
    Set-ItemProperty -Path "IIS:\AppPools\$poolName" -Name "processModel.idleTimeout" -Value ([TimeSpan]::FromMinutes(0))
    
    # Assign pool to site
    Set-ItemProperty -Path "IIS:\Sites\$siteName" -Name "applicationPool" -Value $poolName

    # Configure HTTPS binding
    $cert = Get-ChildItem -Path Cert:\LocalMachine\My | Where-Object { $_.Subject -match "CN=localhost" } | Select-Object -First 1
    if (-not $cert) {
        # Create self-signed certificate if none exists
        $cert = New-SelfSignedCertificate -DnsName "localhost" -CertStoreLocation "Cert:\LocalMachine\My"
    }
    
    New-WebBinding -Name $siteName -Protocol "https" -Port ($port + 1000) -SslFlags 0
    $binding = Get-WebBinding -Name $siteName -Protocol "https"
    $binding.AddSslCertificate($cert.Thumbprint, "my")

    # Open port in Windows Firewall
    $ruleName = "IIS $siteName Port $port"
    if (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue) {
        Remove-NetFirewallRule -DisplayName $ruleName
    }
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $port -Action Allow
}

# Step 3: Configure Application Pools
Write-Output "Configuring application pools..."
foreach ($site in $ports.Keys) {
    $poolName = "$site" + "Pool"
    if (Test-Path "IIS:\AppPools\$poolName") {
        Remove-WebAppPool -Name $poolName
    }
    
    $pool = New-WebAppPool -Name $poolName
    Set-ItemProperty -Path "IIS:\AppPools\$poolName" -Name "managedRuntimeVersion" -Value "v4.0"
    Set-ItemProperty -Path "IIS:\AppPools\$poolName" -Name "startMode" -Value "AlwaysRunning"
    Set-ItemProperty -Path "IIS:\AppPools\$poolName" -Name "processModel.idleTimeout" -Value ([TimeSpan]::FromMinutes(0))
    
    Set-ItemProperty -Path "IIS:\Sites\$site" -Name "applicationPool" -Value $poolName
}

# Step 4: Configure URL Rewrite Rules for API Gateway
Write-Output "Configuring URL rewrite rules..."
try {
    if (-not (Get-Module -Name IISAdministration -ListAvailable)) {
        Write-Output "Installing IIS URL Rewrite Module..."
        Install-Module -Name IISAdministration -Force
    }
    
    # Add URL Rewrite rules for each component
    foreach ($site in $ports.GetEnumerator()) {
        $ruleName = "Redirect_$($site.Key)"
        $pattern = "^$($site.Key)/(.*)$"
        $rewriteUrl = "http://localhost:$($site.Value)/{R:1}"
        
        Add-WebConfigurationProperty -PSPath "MACHINE/WEBROOT/APPHOST" -Filter "system.webServer/rewrite/rules" -Name "." -Value @{
            name = $ruleName
            patternSyntax = "Regular Expressions"
            stopProcessing = "True"
        }
        
        Set-WebConfigurationProperty -PSPath "MACHINE/WEBROOT/APPHOST" -Filter "system.webServer/rewrite/rules/rule[@name='$ruleName']/match" -Name "url" -Value $pattern
        Set-WebConfigurationProperty -PSPath "MACHINE/WEBROOT/APPHOST" -Filter "system.webServer/rewrite/rules/rule[@name='$ruleName']/action" -Name "type" -Value "Rewrite"
        Set-WebConfigurationProperty -PSPath "MACHINE/WEBROOT/APPHOST" -Filter "system.webServer/rewrite/rules/rule[@name='$ruleName']/action" -Name "url" -Value $rewriteUrl
    }
} catch {
    Write-Warning "Failed to configure URL Rewrite rules: $_"
}

# Step 5: Health Check
Write-Output "Performing health check..."
foreach ($site in $ports.GetEnumerator()) {
    $siteName = $site.Key
    $port = $site.Value
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$port" -UseBasicParsing -ErrorAction Stop
        <add name='HippocampusTable' connectionString='Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\Owner\Desktop\database center\SeCuReDmE_Hippocampus.accdb;' />
        <add name='LimbicSystemTable' connectionString='Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\Owner\Desktop\database center\SeCuReDmE_Limbic System.accdb;' />
        <add name='NeuralNetworkTable' connectionString='Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\Owner\Desktop\database center\SeCuReDmE_Neural_network.accdb;' />
        <add name='OccipitalLobeTable' connectionString='Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\Owner\Desktop\database center\SeCuReDmE_Occipital Lobe.accdb;' />
        <add name='PrefrontalCortexTable' connectionString='Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\Owner\Desktop\database center\SeCuReDmE_Prefrontal Cortex.accdb;' />
        <add name='ThalamusTable' connectionString='Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\Owner\Desktop\database center\thalamus_table.sql;' />
    </connectionStrings>
</configuration>
"

# Step 6: Validate Installation and Start Services
Write-Output "Starting and validating services..."
Start-Service W3SVC
Start-Process -FilePath "$IgnitePath\bin\ignite.bat"
Start-Process -FilePath "$MahoutPath\bin\mahout.bat"
try {
    Start-Process -FilePath "$IcebergPath\bin\iceberg.bat" -ErrorAction Stop
} catch {
    Write-Error "Failed to start Iceberg service: $_"
    exit 1
}

Write-Output "Setup Complete. IIS with Apache Ignite, Mahout, and Iceberg is ready for use."
