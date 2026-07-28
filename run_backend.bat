@echo off
cd /d D:\rag_proj
call venv\Scripts\activate.bat
echo ============================================
echo   RAG Platform - FastAPI Backend
echo ============================================
echo.
echo   API Docs : http://localhost:8001/docs
echo   Health   : http://localhost:8001/health
echo.
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
pause
