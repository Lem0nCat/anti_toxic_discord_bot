set VENV_PATH=%CD%\.venv
set SCRIPT_PATH=%CD%\run.py

call "%VENV_PATH%\Scripts\activate.bat"
python "%SCRIPT_PATH%"

call "%VENV_PATH%\Scripts\deactivate.bat"
pause