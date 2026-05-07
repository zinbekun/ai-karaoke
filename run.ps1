# AutoKaraoke 起動スクリプト
# PowerShell で実行: .\run.ps1

$AppDir  = $PSScriptRoot
$uvicorn = Join-Path $AppDir 'venv\Scripts\uvicorn.exe'

if (-not (Test-Path $uvicorn)) {
    Write-Host "セットアップが必要です。先に .\setup.ps1 を実行してください。" -ForegroundColor Red
    Read-Host "Enterキーで終了"
    exit 1
}

Write-Host "`n=== AutoKaraoke 起動 ===" -ForegroundColor Cyan
Write-Host "ブラウザで http://localhost:8000 を開いてください" -ForegroundColor Green
Write-Host "停止するには Ctrl+C を押してください`n" -ForegroundColor Yellow

Set-Location $AppDir
& $uvicorn main:app --host 0.0.0.0 --port 8000 --reload
