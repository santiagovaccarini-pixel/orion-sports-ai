[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExecutable = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExecutable)) {
    throw "Primero ejecutá scripts\windows\Setup-Orion.ps1."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "No se encontró Node.js/npm."
}

$BackendArguments = @(
    "-m", "uvicorn", "backend.app.main:app",
    "--host", "127.0.0.1",
    "--port", "8765"
)

$BackendProcess = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $BackendArguments `
    -WorkingDirectory $ProjectRoot `
    -PassThru `
    -WindowStyle Minimized

try {
    Push-Location $ProjectRoot
    npm exec vite -- --host 127.0.0.1
}
finally {
    Pop-Location
    if (-not $BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id
    }
}
