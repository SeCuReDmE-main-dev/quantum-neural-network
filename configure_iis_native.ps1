# Direct IIS Configuration Script using Microsoft.Web.Administration
Add-Type -Path "${env:windir}\system32\inetsrv\Microsoft.Web.Administration.dll"

Write-Host "Configuring IIS using native API..." -ForegroundColor Cyan

try {
    # Create Server Manager instance
    $manager = New-Object Microsoft.Web.Administration.ServerManager

    # Configure Application Pools
    $pools = @(
        @{
            Name = "PersonaManagerPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "QuantumTensorPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "MindsDBBridgePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "NeuralNetworkCorePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "BasalGangliaPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "CerebrumPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "RightHemispherePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "LeftHemispherePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "CorpusCallosumPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "OccipitalLobePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "ParietalLobePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "TemporalLobePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "FrontalLobePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "GyrusPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "SulcusPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "ThalamusPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "HypothalamusPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "PituitaryGlandPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "PinealGlandPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "LimbicSystemPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "WavePatternPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "CerebellumPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "HippocampusPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "PrefrontalCortexPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "CranialNervesPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "DuraMaterPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "ArachnoidMaterPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "PiaMaterPool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        },
        @{
            Name = "FossaePool"
            RuntimeVersion = "v4.0"
            PipelineMode = "Integrated"
        }
    )

    foreach ($poolConfig in $pools) {
        $pool = $manager.ApplicationPools[$poolConfig.Name]
        if ($pool -eq $null) {
            $pool = $manager.ApplicationPools.Add($poolConfig.Name)
            $pool.ManagedRuntimeVersion = $poolConfig.RuntimeVersion
            $pool.ManagedPipelineMode = $poolConfig.PipelineMode
            Write-Host "Created application pool: $($poolConfig.Name)" -ForegroundColor Green
        } else {
            Write-Host "Application pool already exists: $($poolConfig.Name)" -ForegroundColor Yellow
        }
    }

    # Configure Sites
    $sites = @(
        @{
            Name = "PersonaManager"
            Port = 8083
            Pool = "PersonaManagerPool"
            Path = "C:\inetpub\wwwroot\PersonaManager"
        },
        @{
            Name = "QuantumTensor"
            Port = 8081
            Pool = "QuantumTensorPool"
            Path = "C:\inetpub\wwwroot\QuantumTensor"
        },
        @{
            Name = "MindsDBBridge"
            Port = 8082
            Pool = "MindsDBBridgePool"
            Path = "C:\inetpub\wwwroot\MindsDBBridge"
        },
        @{
            Name = "NeuralNetworkCore"
            Port = 8080
            Pool = "NeuralNetworkCorePool"
            Path = "C:\inetpub\wwwroot\NeuralNetworkCore"
        },
        @{
            Name = "BasalGanglia"
            Port = 8084
            Pool = "BasalGangliaPool"
            Path = "C:\inetpub\wwwroot\BasalGanglia"
        },
        @{
            Name = "Cerebrum"
            Port = 8090
            Pool = "CerebrumPool"
            Path = "C:\inetpub\wwwroot\Cerebrum"
        },
        @{
            Name = "RightHemisphere"
            Port = 8091
            Pool = "RightHemispherePool"
            Path = "C:\inetpub\wwwroot\RightHemisphere"
        },
        @{
            Name = "LeftHemisphere"
            Port = 8092
            Pool = "LeftHemispherePool"
            Path = "C:\inetpub\wwwroot\LeftHemisphere"
        },
        @{
            Name = "CorpusCallosum"
            Port = 8093
            Pool = "CorpusCallosumPool"
            Path = "C:\inetpub\wwwroot\CorpusCallosum"
        },
        @{
            Name = "OccipitalLobe"
            Port = 8094
            Pool = "OccipitalLobePool"
            Path = "C:\inetpub\wwwroot\OccipitalLobe"
        },
        @{
            Name = "ParietalLobe"
            Port = 8095
            Pool = "ParietalLobePool"
            Path = "C:\inetpub\wwwroot\ParietalLobe"
        },
        @{
            Name = "TemporalLobe"
            Port = 8096
            Pool = "TemporalLobePool"
            Path = "C:\inetpub\wwwroot\TemporalLobe"
        },
        @{
            Name = "FrontalLobe"
            Port = 8097
            Pool = "FrontalLobePool"
            Path = "C:\inetpub\wwwroot\FrontalLobe"
        },
        @{
            Name = "Gyrus"
            Port = 8098
            Pool = "GyrusPool"
            Path = "C:\inetpub\wwwroot\Gyrus"
        },
        @{
            Name = "Sulcus"
            Port = 8099
            Pool = "SulcusPool"
            Path = "C:\inetpub\wwwroot\Sulcus"
        },
        @{
            Name = "Thalamus"
            Port = 8100
            Pool = "ThalamusPool"