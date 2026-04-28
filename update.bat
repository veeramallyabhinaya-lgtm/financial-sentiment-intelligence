@echo off
setlocal

echo ============================================================
echo  India Market Sentiment Intelligence — Daily Update
echo ============================================================
echo.

:: Change to project directory (edit this path if needed)
cd /d "D:\financial_sentiment_analysis_project\financial-sentiment-intelligence"

:: Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    echo [1/4] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [1/4] No venv found, using system Python...
)

:: Run the pipeline
echo [2/4] Running sentiment pipeline...
python main.py --limit 50
if %ERRORLEVEL% neq 0 (
    echo ERROR: Pipeline failed. Aborting push.
    pause
    exit /b 1
)

:: Push updated DB to GitHub
echo [3/4] Pushing updated database to GitHub...
git add data/sentiment.db config/calibration.json
git commit -m "Data update %date% %time:~0,5%"
git push
if %ERRORLEVEL% neq 0 (
    echo ERROR: Git push failed. Check your connection.
    pause
    exit /b 1
)

echo [4/4] Done.
echo.
echo Dashboard will reflect changes at your Streamlit URL in ~30 seconds.
echo ============================================================
endlocal