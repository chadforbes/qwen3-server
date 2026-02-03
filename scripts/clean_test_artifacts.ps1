Param(
    [string]$Root = "$(Resolve-Path (Get-Location))"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Cleaning test artifacts under: $Root" -ForegroundColor Cyan

# Remove pytest caches
Get-ChildItem -Path $Root -Recurse -Force -Directory -Filter ".pytest_cache" |
    Where-Object { $_.FullName -notmatch "\\.venv\\" } |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName }

# Remove Python bytecode caches
Get-ChildItem -Path $Root -Recurse -Force -Directory -Filter "__pycache__" |
    Where-Object { $_.FullName -notmatch "\\.venv\\" } |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName }

# Remove stray bytecode files
Get-ChildItem -Path $Root -Recurse -Force -File -Include "*.pyc","*.pyo" |
    Where-Object { $_.FullName -notmatch "\\.venv\\" } |
    ForEach-Object { Remove-Item -Force $_.FullName }

Write-Host "Done." -ForegroundColor Green
