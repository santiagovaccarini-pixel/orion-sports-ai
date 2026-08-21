[CmdletBinding()]
param(
    [switch]$InstallOllama,
    [switch]$DownloadQuickModel
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"
$PythonExecutable = Join-Path $VirtualEnvironment "Scripts\python.exe"

function Resolve-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "No se encontró Python. Instalá Python 3.12 o superior antes de continuar."
}

function Resolve-OllamaExecutable {
    $Command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }
    $Candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            return $Candidate
        }
    }
    return $null
}

Write-Host "Preparando el núcleo local de Orion..." -ForegroundColor Cyan
$PythonCommand = Resolve-PythonCommand

if (-not (Test-Path $PythonExecutable)) {
    if ($PythonCommand.Count -eq 2) {
        & $PythonCommand[0] $PythonCommand[1] -m venv $VirtualEnvironment
    }
    else {
        & $PythonCommand[0] -m venv $VirtualEnvironment
    }
}

& $PythonExecutable -m pip install --upgrade pip
& $PythonExecutable -m pip install -r (Join-Path $ProjectRoot "backend\requirements.txt")

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Warning "No se encontró Node.js 22 o superior. El núcleo quedó instalado, pero falta el entorno de la interfaz."
}
else {
    $NodeMajorVersion = [int]((& node --version).TrimStart("v").Split(".")[0])
    if ($NodeMajorVersion -lt 22) {
        throw "Orion requiere Node.js 22 o superior. La versión detectada es $(& node --version)."
    }
    Push-Location $ProjectRoot
    try {
        npm ci
    }
    finally {
        Pop-Location
    }
}

$OllamaExecutable = Resolve-OllamaExecutable
if ($InstallOllama -and -not $OllamaExecutable) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "No se encontró winget. Instalá Ollama manualmente desde su sitio oficial."
    }
    winget install --id Ollama.Ollama --exact --accept-package-agreements --accept-source-agreements
    $OllamaExecutable = Resolve-OllamaExecutable
}

if ($DownloadQuickModel) {
    if (-not $OllamaExecutable) {
        throw "Ollama todavía no está instalado o requiere reiniciar la terminal."
    }
    & $OllamaExecutable pull qwen3:8b
}

Write-Host "Base de Orion preparada correctamente." -ForegroundColor Green
