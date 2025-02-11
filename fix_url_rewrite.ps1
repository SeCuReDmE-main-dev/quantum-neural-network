# Check and fix IIS URL Rewrite module
$webPlatformInstaller = "${env:ProgramFiles}\Microsoft\Web Platform Installer\WebpiCmd.exe"

# Function to check URL Rewrite module installation
function Test-UrlRewrite {
    $urlRewritePath = "${env:ProgramFiles}\Reference Assemblies\Microsoft\IIS\UrlRewrite.dll"
    return Test-Path $urlRewritePath
}

Write-Host "Checking URL Rewrite module..." -ForegroundColor Cyan

if (-not (Test-UrlRewrite)) {
    Write-Host "URL Rewrite module is not installed." -ForegroundColor Yellow
    
    # Check if Web Platform Installer exists
    if (-not (Test-Path $webPlatformInstaller)) {
        Write-Host "Web Platform Installer is not installed." -ForegroundColor Yellow
        Write-Host "Please install Web Platform Installer from:"
        Write-Host "https://www.microsoft.com/web/downloads/platform.aspx"
        
        # Download Web Platform Installer
        $webpiUrl = "https://download.microsoft.com/download/8/4/9/849DBCF2-DFD9-49F5-9A19-9AEE5B29341A/WebPlatformInstaller_x64_en-US.msi"
        $webpiInstaller = "$env:TEMP\WebPlatformInstaller_x64.msi"
        
        try {
            Invoke-WebRequest -Uri $webpiUrl -OutFile $webpiInstaller
            Write-Host "Installing Web Platform Installer..."
            Start-Process -FilePath msiexec -ArgumentList "/i `"$webpiInstaller`" /qn" -Wait
        }
        catch {
            Write-Error "Failed to download/install Web Platform Installer: $_"
            exit 1
        }
    }
    
    # Install URL Rewrite module
    if (Test-Path $webPlatformInstaller) {
        Write-Host "Installing URL Rewrite module..."
        & $webPlatformInstaller /install /products:UrlRewrite2 /AcceptEula
    }
}
else {
    Write-Host "URL Rewrite module is already installed." -ForegroundColor Green
}

# Configure web.config for URL rewriting
$webConfigContent = @'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <rewrite>
            <rules>
                <rule name="PersonaAPI" stopProcessing="true">
                    <match url="^api/persona/(.*)" />
                    <action type="Rewrite" url="http://localhost:8083/{R:1}" />
                </rule>
                <rule name="QuantumAPI" stopProcessing="true">
                    <match url="^api/quantum/(.*)" />
                    <action type="Rewrite" url="http://localhost:8081/{R:1}" />
                </rule>
                <rule name="NeuralAPI" stopProcessing="true">
                    <match url="^api/neural/(.*)" />
                    <action type="Rewrite" url="http://localhost:8080/{R:1}" />
                </rule>
                <rule name="MindsDBAPI" stopProcessing="true">
                    <match url="^api/mindsdb/(.*)" />
                    <action type="Rewrite" url="http://localhost:8082/{R:1}" />
                </rule>
            </rules>
        </rewrite>
        <handlers>
            <remove name="WebDAV" />
            <remove name="ExtensionlessUrlHandler-Integrated-4.0" />
            <add name="ExtensionlessUrlHandler-Integrated-4.0" path="*." verb="*" type="System.Web.Handlers.TransferRequestHandler" preCondition="integratedMode,runtimeVersionv4.0" />
            <add name="FastCGI-Python" path="*.py" verb="*" type="FastCGI" scriptProcessor="C:\Python310\python.exe|C:\Python310\Scripts\wfastcgi.py" resourceType="Unspecified" />
        </handlers>
        <modules>
            <remove name="WebDAVModule" />
        </modules>
    </system.webServer>
</configuration>
'@

$sites = @(
    "C:\inetpub\wwwroot\PersonaManager",
    "C:\inetpub\wwwroot\QuantumTensor",
    "C:\inetpub\wwwroot\MindsDBBridge",
    "C:\inetpub\wwwroot\NeuralNetworkCore"
)

foreach ($site in $sites) {
    if (Test-Path $site) {
        Set-Content -Path "$site\web.config" -Value $webConfigContent -Force
        Write-Host "Updated web.config in $site" -ForegroundColor Green
    }
    else {
        Write-Host "Site directory not found: $site" -ForegroundColor Yellow
    }
}

Write-Host "`nNext steps:"
Write-Host "1. Run IIS Reset as administrator: iisreset"
Write-Host "2. Verify URL Rewrite module in IIS Manager"
Write-Host "3. Test API endpoints through the rewrite rules"