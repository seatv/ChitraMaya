@echo off
rem packaging/windows/installer/install.cmd
rem Bootstrap run by the ChitraMaya-install.exe SFX after it unpacks its
rem payload (this file + install.ps1 + 7zr.exe) to a temp folder.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
exit /b %ERRORLEVEL%
