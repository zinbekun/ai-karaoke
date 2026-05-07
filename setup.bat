@echo off
chcp 65001 > nul
echo === AutoKaraoke セットアップ ===
echo.

REM Python の確認
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3 が見つかりません。
    echo 以下からインストールしてください:
    echo   https://www.python.org/downloads/
    echo   または: winget install Python.Python.3.12
    echo.
    echo インストール後、このスクリプトを再実行してください。
    pause
    exit /b 1
)

echo Python 確認済み:
python --version

REM 仮想環境の作成
if not exist "%~dp0venv" (
    echo.
    echo 仮想環境を作成中...
    python -m venv "%~dp0venv"
    echo 仮想環境を作成しました。
) else (
    echo 仮想環境は既に存在します。
)

REM pip のアップグレード
echo.
echo pip をアップグレード中...
"%~dp0venv\Scripts\pip.exe" install --upgrade pip --quiet

REM 依存ライブラリのインストール
echo.
echo 依存ライブラリをインストール中...
echo (librosa などのインストールに数分かかります)
"%~dp0venv\Scripts\pip.exe" install -r "%~dp0requirements.txt"

echo.
echo === セットアップ完了！===
echo アプリを起動するには run.bat をダブルクリックしてください。
pause
