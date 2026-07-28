@echo off
cd /d D:\rag_proj
call venv\Scripts\activate.bat
echo ============================================
echo   RAG Platform - Streamlit Frontend
echo ============================================
echo.
echo   Frontend : http://localhost:8501
echo.
streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1
pause
