[CmdletBinding()]
param(
    [switch]$Cloud,
    [switch]$LocalLegacy,
    [string]$CloudUrl = "https://orion-core-prototype.onrender.com"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExecutable = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RuntimeDirectory = Join-Path $ProjectRoot ".orion-runtime"
$BackendOutputLog = Join-Path $RuntimeDirectory "backend-output.log"
$BackendErrorLog = Join-Path $RuntimeDirectory "backend-error.log"
$BackendHealthUrl = "http://127.0.0.1:8765/api/v1/health"

if ($Cloud -and $LocalLegacy) {
    throw "No se pueden combinar -Cloud y -LocalLegacy. Orion Cloud es el modo normal."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "No se encontró Node.js/npm."
}

New-Item -ItemType Directory -Path $RuntimeDirectory -Force | Out-Null

function Start-OrionFrontend {
    Push-Location $ProjectRoot
    try {
        # Desactiva los atajos interactivos de Vite/Cloudflare. Así, pegar un
        # comando en esta terminal no puede abrir un túnel público por accidente.
        $env:CI = "1"
        npm exec vite -- --host 127.0.0.1
    }
    finally {
        Pop-Location
    }
}

# Orion Cloud es el único modo normal. -Cloud se conserva únicamente para que los
# accesos directos/comandos anteriores sigan funcionando. El backend local con
# Ollama/Qwen requiere -LocalLegacy de forma explícita.
if (-not $LocalLegacy) {
    if (-not $env:ORION_API_KEY) {
        throw "Falta ORION_API_KEY en esta terminal. Definila antes de iniciar Orion Cloud."
    }

    $NormalizedCloudUrl = $CloudUrl.TrimEnd("/")
    $env:NEXT_PUBLIC_ORION_API_URL = "$NormalizedCloudUrl/api/v1"
    $env:NEXT_PUBLIC_ORION_API_KEY = $env:ORION_API_KEY

    Write-Host "Orion conectado al núcleo cloud: $NormalizedCloudUrl" -ForegroundColor Green
    Write-Host "Motor activo: Cloudflare Workers AI / gpt-oss. El backend local Qwen no se inicia." -ForegroundColor Green
    Write-Host "Modo de prueba local de la interfaz: no publiques esta compilación porque la API key queda disponible en el navegador local." -ForegroundColor Yellow
    Start-OrionFrontend
    exit 0
}

Write-Warning "Iniciando compatibilidad LOCAL LEGACY (Ollama/Qwen). Este no es el Orion canónico actual."

if (-not (Test-Path $PythonExecutable)) {
    throw "Primero ejecutá scripts\windows\Setup-Orion.ps1."
}

$BackendArguments = @(
    "-m", "uvicorn", "backend.app.main:app",
    "--host", "127.0.0.1",
    "--port", "8765"
)

function Wait-OrionBackend {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 20
    )

    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $Deadline) {
        if ($Process.HasExited) {
            $LogHint = "Revisá el registro: $BackendErrorLog"
            if (Test-Path $BackendErrorLog) {
                $RecentErrors = (Get-Content $BackendErrorLog -Tail 15) -join [Environment]::NewLine
                if ($RecentErrors) {
                    $LogHint = "$LogHint$([Environment]::NewLine)$RecentErrors"
                }
            }
            throw "El núcleo local legacy de Orion se cerró durante el inicio.$([Environment]::NewLine)$LogHint"
        }

        try {
            $Health = Invoke-RestMethod -Uri $BackendHealthUrl -TimeoutSec 1
            if ($Health.status -eq "ok") {
                return
            }
        }
        catch {
            # El servidor puede tardar unos segundos en aceptar conexiones.
        }

        Start-Sleep -Milliseconds 250
    }

    throw "El núcleo local legacy no respondió a tiempo. Revisá: $BackendErrorLog"
}

$BackendProcess = $null

try {
    $env:ORION_MODEL_PROVIDER = "ollama"
    $BackendProcess = Start-Process `
        -FilePath $PythonExecutable `
        -ArgumentList $BackendArguments `
        -WorkingDirectory $ProjectRoot `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $BackendOutputLog `
        -RedirectStandardError $BackendErrorLog

    Wait-OrionBackend -Process $BackendProcess
    Write-Host "Núcleo local legacy iniciado en segundo plano." -ForegroundColor Yellow
    Start-OrionFrontend
}
finally {
    if ($BackendProcess -and -not $BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id
    }
}
