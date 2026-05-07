@echo off
chcp 65001 > nul
title AutoKaraoke 公開モード

echo.
echo  ========================================
echo    AI Karaoke  -  インターネット公開
echo  ========================================
echo.

REM ─── 環境変数 PATH を最新に ────────────────────────────
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable(\"PATH\",\"Machine\")"') do set "MACHINE_PATH=%%i"
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable(\"PATH\",\"User\")"') do set "USER_PATH=%%i"
set "PATH=%MACHINE_PATH%;%USER_PATH%"

REM ─── セットアップ確認 ──────────────────────────────────
if not exist "%~dp0venv\Scripts\uvicorn.exe" (
    echo セットアップが必要です。先に start.bat を実行してください。
    pause & exit /b 1
)

where ngrok >nul 2>&1
if errorlevel 1 (
    echo ngrok が見つかりません。インストールしてください:
    echo   winget install ngrok.ngrok
    pause & exit /b 1
)

REM ─── ngrok 認証トークン確認 ───────────────────────────
ngrok config check >nul 2>&1
if errorlevel 1 (
    echo.
    echo  !! ngrok の初回セットアップが必要です !!
    echo.
    echo  以下の手順を行ってください:
    echo.
    echo  1. https://ngrok.com にアクセスして無料アカウントを作成
    echo  2. ログイン後、左メニューの「Your Authtoken」をクリック
    echo  3. 表示されたトークンをコピー
    echo  4. 以下のコマンドを実行（YOUR_TOKENを貼り付け）:
    echo.
    echo     ngrok config add-authtoken YOUR_TOKEN
    echo.
    echo  5. その後、このファイルをもう一度ダブルクリック
    echo.
    start https://ngrok.com/signup
    pause & exit /b 1
)

REM ─── サーバー起動（バックグラウンド） ────────────────
echo [1/2] カラオケサーバーを起動中...
start "" "%~dp0venv\Scripts\uvicorn.exe" main:app --host 127.0.0.1 --port 8000
timeout /t 3 /nobreak >nul

REM ─── ngrok トンネル起動 ───────────────────────────────
echo [2/2] インターネットに公開中...
echo.
echo  ┌─────────────────────────────────────────────────────────┐
echo  │                                                         │
echo  │  起動したら「Forwarding」行のURLをスマホ等で開いてください  │
echo  │  例: https://xxxx-xxxx.ngrok-free.app                  │
echo  │                                                         │
echo  │  ★ 無料の固定URLを取得する方法:                         │
echo  │    ngrok.com → 左メニュー「Domains」→「New Domain」      │
echo  │    その後: ngrok http --domain=あなたのドメイン 8000     │
echo  │                                                         │
echo  │  終了するには Ctrl+C を押してください                    │
echo  │                                                         │
echo  └─────────────────────────────────────────────────────────┘
echo.

ngrok http 8000
pause
