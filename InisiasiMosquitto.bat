@echo off
:: Check for permissions
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"

:: If error flag set, we do not have admin.
if '%errorlevel%' NEQ '0' (
    echo Requesting administrative privileges...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"

    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "C:\Program Files\mosquitto"
    
    :: Check if mosquitto.conf exists
    if not exist "mosquitto.conf" (
        echo Creating new mosquitto.conf file...
        echo listener 1883 0.0.0.0> mosquitto.conf
        echo allow_anonymous true>> mosquitto.conf
        goto StartMosquitto
    )

    :: Check if required settings exist in mosquitto.conf
    findstr /C:"listener 1883 0.0.0.0" mosquitto.conf >nul
    if errorlevel 1 (
        echo Adding listener configuration...
        echo listener 1883 0.0.0.0>> mosquitto.conf
    )

    findstr /C:"allow_anonymous true" mosquitto.conf >nul
    if errorlevel 1 (
        echo Adding anonymous access configuration...
        echo allow_anonymous true>> mosquitto.conf
    )

:StartMosquitto
    :: Kill any existing mosquitto processes
    taskkill /F /IM mosquitto.exe /T > nul 2>&1
    
    :: Wait a moment to ensure port is released
    timeout /t 2 > nul
    
    :: Run Mosquitto with verbose output
    echo Starting Mosquitto in verbose mode...
    echo Press Ctrl+C to stop the server cleanly
    echo NOTE: DO NOT click X to close this window, use Ctrl+C instead
    
    :: Start Mosquitto
    mosquitto -c mosquitto.conf -v
    
    :: When Mosquitto exits (either through Ctrl+C or window close)
    echo.
    echo Cleaning up...
    taskkill /F /IM mosquitto.exe /T > nul 2>&1
    timeout /t 2 > nul
    
    echo Press any key to exit...
    pause > nul