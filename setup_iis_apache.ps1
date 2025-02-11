# Elevate to admin if not already
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {    
    $arguments = "& '" + $myinvocation.mycommand.definition + "'"
    Start-Process powershell -Verb runAs -ArgumentList $arguments
    Break
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
    
    # Create site directory
    New-Item -ItemType Directory -Path $sitePath -Force
    
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

    # Create or update website
    if (Get-Website -Name $siteName) {
        Remove-Website -Name $siteName
    }
    New-Website -Name $siteName -PhysicalPath $sitePath -Port $port -Force
    
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
    # Check if URL Rewrite Module is installed
    if (-not (Test-Path "$env:ProgramFiles\IIS\Application Request Routing")) {
        Write-Output "Downloading and installing URL Rewrite Module..."
        $urlRewriteInstaller = "$env:TEMP\urlrewrite.msi"
        Invoke-WebRequest -Uri "https://download.microsoft.com/download/1/2/8/128E2E22-C1B9-44A4-BE2A-5859ED1D4592/rewrite_amd64_en-US.msi" -OutFile $urlRewriteInstaller
        Start-Process -FilePath "msiexec.exe" -ArgumentList "/i `"$urlRewriteInstaller`" /quiet /norestart" -Wait
    }
} catch {
    Write-Warning "Failed to install URL Rewrite Module. Some functionality may be limited."
    Write-Warning $_.Exception.Message
}

# Step 5: Health Check
Write-Output "Performing health check..."
foreach ($site in $ports.GetEnumerator()) {
    $siteName = $site.Key
    $port = $site.Value
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$port" -UseBasicParsing -ErrorAction Stop
        Write-Output "$siteName is running on port $port. Status: $($response.StatusCode)"
    } catch {
        Write-Warning "$siteName health check failed on port $port"
        Write-Warning $_.Exception.Message
    }
}

Write-Output "IIS setup complete. Please check the above output for any warnings or errors."

# Step 2: Download and Configure Apache Ignite
Write-Output "Downloading and configuring Apache Ignite..."
$IgniteVersion = "2.14.0"
$IgniteDownloadURL = "https://archive.apache.org/dist/ignite/$IgniteVersion/apache-ignite-$IgniteVersion-bin.zip"
$IgnitePath = "C:\Apache\Ignite"
Invoke-WebRequest -Uri $IgniteDownloadURL -OutFile "$IgnitePath\apache-ignite.zip"
Expand-Archive -Path "$IgnitePath\apache-ignite.zip" -DestinationPath $IgnitePath

# Configure Ignite
$IgniteConfigFile = "$IgnitePath\config\default-config.xml"
Set-Content -Path $IgniteConfigFile -Value "
<bean id='ignite.cfg' class='org.apache.ignite.configuration.IgniteConfiguration'>
    <property name='peerClassLoadingEnabled' value='true'/>
</bean>
"

# Step 3: Download and Configure Apache Mahout
Write-Output "Downloading and configuring Apache Mahout..."
$MahoutVersion = "14.1.0"
$MahoutDownloadURL = "https://archive.apache.org/dist/mahout/$MahoutVersion/apache-mahout-distribution-$MahoutVersion.zip"
$MahoutPath = "C:\Apache\Mahout"
Invoke-WebRequest -Uri $MahoutDownloadURL -OutFile "$MahoutPath\apache-mahout.zip"
Expand-Archive -Path "$MahoutPath\apache-mahout.zip" -DestinationPath $MahoutPath

# Step 4: Download and Configure Apache Iceberg
Write-Output "Downloading and configuring Apache Iceberg..."
$IcebergVersion = "0.12.0"
$IcebergDownloadURL = "https://archive.apache.org/dist/iceberg/$IcebergVersion/apache-iceberg-$IcebergVersion-bin.zip"
$IcebergPath = "C:\Apache\Iceberg"
New-Item -ItemType Directory -Force -Path $IcebergPath
Invoke-WebRequest -Uri $IcebergDownloadURL -OutFile "$IcebergPath\apache-iceberg.zip"
Expand-Archive -Path "$IcebergPath\apache-iceberg.zip" -DestinationPath $IcebergPath

# Step 5: Integrate Components with IIS
Write-Output "Configuring integration between IIS, Ignite, Mahout, and Iceberg..."
# Example configuration for linking Ignite and Mahout
Set-Content -Path "$sitePath\Web.config" -Value "
<configuration>
    <appSettings>
        <add key='ApacheIgnitePath' value='$IgnitePath'/>
        <add key='ApacheMahoutPath' value='$MahoutPath'/>
        <add key='ApacheIcebergPath' value='$IcebergPath'/>
        <add key='DatabaseCenterPath' value='C:\Users\Owner\Desktop\database center'/>
    </appSettings>
    <connectionStrings>
        <add name='CerebellumTable' connectionString='Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\Owner\Desktop\database center\SeCuReDmE_Cerebellum.accdb;' />
        <add name='CerebrumTable' connectionString='Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\Owner\Desktop\database center\SeCuReDmE_Cerebrum.accdb;' />
        <add name='CreateBrainTables' connectionString='Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\Owner\Desktop\database center\create_brain_tables.sql;' />
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
