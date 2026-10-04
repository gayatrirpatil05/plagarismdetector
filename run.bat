@echo off
REM Local DEVELOPMENT launcher (auto-reload, http cookies). Not for production.
echo.
echo  SemanticEye ^| local development server
echo  -------------------------------------------
echo.

if not exist .venv (
    echo  Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

set APP_ENV=development
echo  Open http://127.0.0.1:8000   (Ctrl+C to stop)
echo.
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
