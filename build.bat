@echo off
rem Builds PickTheLock.exe into the repository root, next to config.json.
rem Uses the project's virtual environment (.venv) when it exists.
cd /d "%~dp0"
set PYTHON=python
if exist .venv\Scripts\python.exe set PYTHON=.venv\Scripts\python.exe
%PYTHON% -m PyInstaller --noconfirm --onefile --console --name PickTheLock ^
    --hidden-import dxcam --hidden-import comtypes ^
    --distpath . --workpath build --specpath build ^
    picklock.py
