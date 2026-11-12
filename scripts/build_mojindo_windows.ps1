# Build Mojindo-X-Anylabeling Windows CPU portable package (onedir).
# Usage (from repo root, with venv activated):
#   powershell -ExecutionPolicy Bypass -File scripts/build_mojindo_windows.ps1

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = (Get-Command python).Source
}

$PyInstaller = Join-Path $Root ".venv\Scripts\pyinstaller.exe"
if (-not (Test-Path $PyInstaller)) {
    $PyInstaller = (Get-Command pyinstaller).Source
}

$Spec = Join-Path $Root "packaging\pyinstaller\specs\mojindo-x-anylabeling-win-cpu.spec"
$IconSrc = "C:\WorkSpace\env\mojindo\mojindo.png"
$IconIco = Join-Path $Root "anylabeling\resources\images\mojindo.ico"
$IconPng = Join-Path $Root "anylabeling\resources\images\icon.png"

Write-Host "Refreshing Mojindo icons..."
& $Python -c @"
from PIL import Image
from pathlib import Path
src = Path(r'$IconSrc')
if not src.is_file():
    raise SystemExit(f'Icon not found: {src}')
im = Image.open(src).convert('RGBA')
out_ico = Path(r'$IconIco')
out_png = Path(r'$IconPng')
sizes = [(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)]
imgs = [im.resize(s, Image.Resampling.LANCZOS) for s in sizes]
imgs[-1].save(out_ico, format='ICO', sizes=[(i.width, i.height) for i in imgs])
im.resize((256, 256), Image.Resampling.LANCZOS).save(out_png)
Path(r'$Root\anylabeling\resources\images\icon.ico').write_bytes(out_ico.read_bytes())
print('icons ok')
"@

$env:X_ANYLABELING_ROOT = $Root
$env:X_ANYLABELING_PACKAGE_MODE = "onedir"
$env:X_ANYLABELING_DEVICE = "CPU"

Write-Host "Building with PyInstaller..."
& $PyInstaller --noconfirm --clean $Spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$AppInfo = Get-Content (Join-Path $Root "anylabeling\app_info.py") -Raw
if ($AppInfo -match '__version__\s*=\s*"([^"]+)"') { $Version = $Matches[1] } else { $Version = "unknown" }
if ($AppInfo -match '__appname__\s*=\s*"([^"]+)"') { $AppName = $Matches[1] } else { $AppName = "Mojindo-X-Anylabeling" }

$DistDir = Join-Path $Root "dist\$AppName"
$ExePath = Join-Path $DistDir "$AppName.exe"
if (-not (Test-Path $ExePath)) {
    throw "Expected executable not found: $ExePath"
}

$ZipName = "$AppName-v$Version-Windows-CPU.zip"
$ZipPath = Join-Path $Root "dist\$ZipName"
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }

Write-Host "Creating zip: $ZipPath"
Compress-Archive -Path $DistDir -DestinationPath $ZipPath -CompressionLevel Optimal

Write-Host ""
Write-Host "Build complete."
Write-Host "  Folder: $DistDir"
Write-Host "  Exe:    $ExePath"
Write-Host "  Zip:    $ZipPath"
Write-Host "Double-click $AppName.exe inside the folder to launch."
