@echo off
chcp 65001 > nul
echo === AutoKaraoke 起動 ===
echo.

if not exist "%~dp0venv\Scripts\uvicorn.exe" (
    echo セットアップが必要です。setup.bat を先に実行してください。
    pause
    exit /b 1
)

echo ブラウザで http://localhost:8000 を開いてください
echo 停止するには Ctrl+C を押してください
echo.

cd /d "%~dp0"
venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8000 --reload
pause
