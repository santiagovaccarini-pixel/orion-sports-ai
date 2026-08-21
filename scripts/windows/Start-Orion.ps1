[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExecutable = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RuntimeDirectory = Join-Path $ProjectRoot ".orion-runtime"
$BackendOutputLog = Join-Path $RuntimeDirectory "backend-output.log"
$BackendErrorLog = Join-Path $RuntimeDirectory "backend-error.log"
$BackendHealthUrl = "http://127.0.0.1:8765/api/v1/health"

if (-not (Test-Path $PythonExecutable)) {
    throw "Primero ejecutá scripts\windows\Setup-Orion.ps1."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "No se encontró Node.js/npm."
}

New-Item -ItemType Directory -Path $RuntimeDirectory -Force | Out-Null

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
            throw "El núcleo local de Orion se cerró durante el inicio.$([Environment]::NewLine)$LogHint"
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

    throw "El núcleo local de Orion no respondió a tiempo. Revisá: $BackendErrorLog"
}

$BackendProcess = $null

try {
    $BackendProcess = Start-Process `
        -FilePath $PythonExecutable `
        -ArgumentList $BackendArguments `
        -WorkingDirectory $ProjectRoot `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $BackendOutputLog `
        -RedirectStandardError $BackendErrorLog

    Wait-OrionBackend -Process $BackendProcess
    Write-Host "Núcleo local iniciado en segundo plano." -ForegroundColor Green

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
finally {
    if ($BackendProcess -and -not $BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id
    }
}
