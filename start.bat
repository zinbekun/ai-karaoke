@echo off
chcp 65001 > nul
title AutoKaraoke

echo.
echo  ========================================
echo    AI Karaoke  -  自動セットアップ起動
echo  ========================================
echo.

REM ─── STEP 1: Python 確認 ───────────────────────────────
echo [1/3] Python を確認しています...

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  !! Python がインストールされていません !!
    echo.
    echo  以下の手順でインストールしてください:
    echo.
    echo  1. 下のURLをブラウザで開く
    echo     https://www.python.org/downloads/
    echo.
    echo  2. 黄色い「Download Python 3.x.x」ボタンをクリック
    echo.
    echo  3. ダウンロードしたファイルを実行
    echo     ★ 必ず「Add python.exe to PATH」に チェック ★
    echo.
    echo  4. インストール完了後、このファイルをもう一度ダブルクリック
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo     OK!
echo.

REM ─── STEP 2: 仮想環境 + ライブラリ ───────────────────────
if not exist "%~dp0venv\Scripts\uvicorn.exe" (
    echo [2/3] ライブラリをインストールしています...
    echo       ^(初回のみ。数分かかります^)
    echo.

    if not exist "%~dp0venv" (
        python -m venv "%~dp0venv"
        if errorlevel 1 (
            echo !! 仮想環境の作成に失敗しました !!
            pause
            exit /b 1
        )
    )

    "%~dp0venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    "%~dp0venv\Scripts\python.exe" -m pip install fastapi "uvicorn[standard]" python-multipart librosa numpy soundfile

    if errorlevel 1 (
        echo.
        echo  !! インストールに失敗しました !!
        pause
        exit /b 1
    )

    echo.
    echo     インストール完了!
    echo.
) else (
    echo [2/3] ライブラリは導入済みです。スキップ
    echo.
)

REM ─── STEP 3: サーバー起動 ────────────────────────────────
echo [3/3] アプリを起動しています...
echo.
echo  ┌─────────────────────────────────────────┐
echo  │                                         │
echo  │   ブラウザが自動で開きます              │
echo  │   開かない場合は以下をブラウザで開く:   │
echo  │                                         │
echo  │     http://localhost:8000               │
echo  │                                         │
echo  │   終了するには このウィンドウを閉じる   │
echo  │                                         │
echo  └─────────────────────────────────────────┘
echo.

REM ブラウザを先に起動（uvicorn が起動する間に開く）
start "" "http://localhost:8000"

cd /d "%~dp0"

REM ffmpeg が PATH に入るよう環境変数を更新してから起動
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable(\"PATH\",\"Machine\")"') do set "MACHINE_PATH=%%i"
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable(\"PATH\",\"User\")"') do set "USER_PATH=%%i"
set "PATH=%MACHINE_PATH%;%USER_PATH%"

"%~dp0venv\Scripts\uvicorn.exe" main:app --host 127.0.0.1 --port 8000
pause
