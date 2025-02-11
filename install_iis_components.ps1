# Run this script as Administrator to install missing IIS components
Write-Host "Installing IIS components..." -ForegroundColor Cyan
dism /online /enable-feature /featurename:IIS-ManagementScriptingTools /all
dism /online /enable-feature /featurename:IIS-NetFxExtensibility45 /all
dism /online /enable-feature /featurename:IIS-ISAPIExtensions /all
dism /online /enable-feature /featurename:IIS-ISAPIFilter /all
dism /online /enable-feature /featurename:IIS-ASPNET45 /all
dism /online /enable-feature /featurename:IIS-ApplicationInit /all
dism /online /enable-feature /featurename:IIS-WindowsAuthentication /all
dism /online /enable-feature /featurename:IIS-DigestAuthentication /all
dism /online /enable-feature /featurename:IIS-BasicAuthentication /all

Write-Host "
Installation complete. Please run validate_iis_setup.ps1 again to verify." -ForegroundColor Green
