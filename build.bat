@echo off
rem Builds PickTheLock.exe into the repository root, next to config.json.
cd /d "%~dp0"
pyinstaller --noconfirm --onefile --console --name PickTheLock ^
    --hidden-import dxcam --hidden-import comtypes ^
    --distpath . --workpath build --specpath build ^
    picklock.py
