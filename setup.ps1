# AutoKaraoke セットアップスクリプト
# PowerShell で実行: .\setup.ps1

$ErrorActionPreference = 'Stop'
$AppDir = $PSScriptRoot

Write-Host "`n=== AutoKaraoke セットアップ ===" -ForegroundColor Cyan

# 1. Python の確認
$pythonCmd = $null
foreach ($cmd in @('python3', 'python')) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match 'Python 3\.\d+') {
            $pythonCmd = $cmd
            Write-Host "Python 確認済み: $ver" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $pythonCmd) {
    Write-Host "`nPython 3 が見つかりません。" -ForegroundColor Red
    Write-Host "以下のいずれかの方法でインストールしてください:" -ForegroundColor Yellow
    Write-Host "  1. https://www.python.org/downloads/ からダウンロード"
    Write-Host "  2. winget install Python.Python.3.12"
    Write-Host "  3. Microsoft Store から Python 3.12 をインストール"
    Write-Host "`nインストール後、このスクリプトを再実行してください。"
    Read-Host "Enterキーで終了"
    exit 1
}

# 2. 仮想環境の作成
$venvDir = Join-Path $AppDir 'venv'
if (-not (Test-Path $venvDir)) {
    Write-Host "`n仮想環境を作成中..." -ForegroundColor Yellow
    & $pythonCmd -m venv $venvDir
    Write-Host "仮想環境を作成しました: $venvDir" -ForegroundColor Green
} else {
    Write-Host "仮想環境は既に存在します: $venvDir" -ForegroundColor Green
}

# 3. pip のアップグレード
$pip = Join-Path $venvDir 'Scripts\pip.exe'
Write-Host "`npip をアップグレード中..." -ForegroundColor Yellow
& $pip install --upgrade pip --quiet

# 4. 依存ライブラリのインストール
Write-Host "`n依存ライブラリをインストール中..." -ForegroundColor Yellow
Write-Host "（librosa や numpy などのインストールに数分かかります）"
& $pip install -r (Join-Path $AppDir 'requirements.txt')

Write-Host "`n=== セットアップ完了! ===" -ForegroundColor Green
Write-Host "アプリを起動するには: .\run.ps1" -ForegroundColor Cyan
Read-Host "Enterキーで終了"
