# Step 1: Install IIS and Enable Necessary Features
Write-Output "Installing IIS and enabling necessary features..."
Install-WindowsFeature -Name Web-Server, Web-Mgmt-Tools, Web-Scripting-Tools, Web-Asp-Net45

# Configure IIS Site
$SiteName = "NeuralNetworkCore"
$SitePath = "C:\inetpub\wwwroot\$SiteName"
New-Item -ItemType Directory -Path $SitePath -Force
New-Website -Name $SiteName -PhysicalPath $SitePath -Port 8080 -Force

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
Set-Content -Path "$SitePath\Web.config" -Value "
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
