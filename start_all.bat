@echo off
cd /d %~dp0
call venv\Scripts\activate.bat

echo ============================================
echo   RAG Platform - Full Startup
echo ============================================
echo.

echo [1/3] Starting Docker services (Milvus + Redis)...
docker-compose up -d
if %errorlevel% neq 0 (
    echo [ERROR] Docker failed. Please start Docker Desktop first.
    pause
    exit /b 1
)
echo        Waiting for Milvus to be ready...
timeout /t 15 /nobreak >nul
echo        Docker services are up.
echo.

echo [2/3] Starting FastAPI backend (port 8001)...
start "RAG-Backend" cmd /c "cd /d D:\rag_proj && venv\Scripts\activate.bat && uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload"
echo        Backend starting... http://localhost:8001/docs
echo.

echo [3/3] Starting Streamlit frontend (port 8501)...
start "RAG-Frontend" cmd /c "cd /d D:\rag_proj && venv\Scripts\activate.bat && streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1"
echo        Frontend starting... http://localhost:8501
echo.

echo ============================================
echo   Startup Complete!
echo   Frontend : http://localhost:8501
echo   API Docs : http://localhost:8001/docs
echo ============================================
echo.
echo   Close the popup windows to stop services.
echo.
pause
