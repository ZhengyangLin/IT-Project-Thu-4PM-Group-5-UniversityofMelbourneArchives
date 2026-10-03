@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "UMA_LAUNCH_ROOT=%~dp0"

set "UMA_LAUNCH_PYTHON=%UMA_PYTHON%"
set "UMA_LAUNCH_CONDA="
set "UMA_LAUNCH_ACTIVATE="
set "UMA_LAUNCH_NODE="
set "UMA_LAUNCH_CHECK="

:parse_args
if "%~1"=="" goto prepare
if /i "%~1"=="-Check" (
    set "UMA_LAUNCH_CHECK=1"
    shift
    goto parse_args
)
if /i "%~1"=="-PythonExe" (
    if "%~2"=="" (
        echo Startup failed: -PythonExe requires a Python executable path.
        goto failed
    )
    set "UMA_LAUNCH_PYTHON=%~2"
    shift
    shift
    goto parse_args
)
echo Startup failed: Unknown option "%~1".
echo Usage: start.bat [-Check] [-PythonExe "C:\path\to\python.exe"]
goto failed

:prepare
cd /d "%UMA_LAUNCH_ROOT%"
if errorlevel 1 goto failed
if not defined UMA_LAUNCH_PYTHON call :find_python
if not defined UMA_LAUNCH_PYTHON (
    echo Startup failed: Python not found. Set UMA_PYTHON or pass -PythonExe.
    goto failed
)
if not exist "%UMA_LAUNCH_PYTHON%" (
    echo Startup failed: Python executable not found: "%UMA_LAUNCH_PYTHON%"
    goto failed
)
for %%P in ("%UMA_LAUNCH_PYTHON%") do (
    set "UMA_LAUNCH_PYTHON=%%~fP"
    set "UMA_LAUNCH_PREFIX=%%~dpP"
)
echo Python: "%UMA_LAUNCH_PYTHON%"
if exist "%UMA_LAUNCH_PREFIX%conda-meta\" (
    call :find_conda
    if not defined UMA_LAUNCH_ACTIVATE (
        echo Startup failed: Conda activation script not found. Set CONDA_EXE to your Conda installation's conda.exe.
        goto failed
    )
)
rem Activate once in this launcher; service windows inherit the native DLL paths.
for %%P in ("%UMA_LAUNCH_PREFIX%.") do set "UMA_LAUNCH_PREFIX=%%~fP"
if defined UMA_LAUNCH_ACTIVATE (
    call "%UMA_LAUNCH_ACTIVATE%" activate "%UMA_LAUNCH_PREFIX%"
    if errorlevel 1 goto failed
)

for /f "delims=" %%N in ('where node.exe 2^>nul') do if not defined UMA_LAUNCH_NODE set "UMA_LAUNCH_NODE=%%N"
if not defined UMA_LAUNCH_NODE (
    echo Startup failed: Node.js is missing from PATH.
    goto failed
)
"%UMA_LAUNCH_NODE%" -e "const [major, minor] = process.versions.node.split('.').map(Number); if (!((major === 22 && minor >= 18) || (major === 24 && minor >= 12) || major > 24)) { console.error('Node.js must be 22.18+ (22.x) or 24.12+.'); process.exit(1); }"
if errorlevel 1 goto failed
if not exist "%UMA_LAUNCH_ROOT%front-end\node_modules\vite\bin\vite.js" (
    echo Startup failed: Run "yarn install --frozen-lockfile" inside front-end first.
    goto failed
)
"%UMA_LAUNCH_PYTHON%" -c "import fastapi, uvicorn, multipart"
if errorlevel 1 (
    echo Startup failed: Check backend dependencies in the Python environment printed above.
    goto failed
)
for %%P in (8000 5173) do (
    "%UMA_LAUNCH_NODE%" -e "const net = require('node:net'); const port = Number(process.argv[1]); const server = net.createServer(); server.on('error', () => { console.error('Port ' + port + ' is unavailable. Close the existing service, then retry.'); process.exit(1); }); server.listen({host: '127.0.0.1', port, exclusive: true}, () => server.close());" %%P
    if errorlevel 1 goto failed
)
echo Startup checks passed.
if defined UMA_LAUNCH_CHECK exit /b 0

rem Run executables directly: no active batch job to prompt on Ctrl+C.
rem /k keeps each visible console and its logs after the service stops.
start "UMA Backend - Ctrl+C to stop" /D "%UMA_LAUNCH_ROOT%." "%ComSpec%" /d /s /k ""%UMA_LAUNCH_PYTHON%" -u serve.py --host 127.0.0.1 --port 8000"
if errorlevel 1 goto failed
start "UMA Frontend - Ctrl+C to stop" /D "%UMA_LAUNCH_ROOT%front-end" "%ComSpec%" /d /s /k ""%UMA_LAUNCH_NODE%" node_modules\vite\bin\vite.js --host 127.0.0.1 --port 5173 --strictPort"
if errorlevel 1 goto failed
echo Opened Backend and Frontend CMD windows.
echo Once both are ready, visit http://127.0.0.1:5173
echo Stop each service with Ctrl+C, or close both service windows.
exit /b 0

:find_python
for %%P in ("%UMA_LAUNCH_ROOT%.venv\Scripts\python.exe" "%UMA_LAUNCH_ROOT%venv\Scripts\python.exe") do if not defined UMA_LAUNCH_PYTHON if exist "%%~P" set "UMA_LAUNCH_PYTHON=%%~fP"
if not defined UMA_LAUNCH_PYTHON if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "UMA_LAUNCH_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined UMA_LAUNCH_PYTHON if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "UMA_LAUNCH_PYTHON=%CONDA_PREFIX%\python.exe"
for %%P in ("%USERPROFILE%\miniconda3\envs\uma\python.exe" "%USERPROFILE%\anaconda3\envs\uma\python.exe") do if not defined UMA_LAUNCH_PYTHON if exist "%%~P" set "UMA_LAUNCH_PYTHON=%%~fP"
if defined UMA_LAUNCH_PYTHON exit /b 0
for /f "delims=" %%P in ('where python.exe 2^>nul') do if not defined UMA_LAUNCH_PYTHON set "UMA_LAUNCH_PYTHON=%%P"
if defined UMA_LAUNCH_PYTHON exit /b 0
for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "UMA_LAUNCH_PYTHON=%%P"
exit /b 0

:find_conda
if defined CONDA_EXE if exist "%CONDA_EXE%" set "UMA_LAUNCH_CONDA=%CONDA_EXE%"
for %%C in ("%UMA_LAUNCH_PREFIX%Scripts\conda.exe" "%UMA_LAUNCH_PREFIX%..\..\Scripts\conda.exe" "%USERPROFILE%\miniconda3\Scripts\conda.exe" "%USERPROFILE%\anaconda3\Scripts\conda.exe") do if not defined UMA_LAUNCH_CONDA if exist "%%~C" set "UMA_LAUNCH_CONDA=%%~fC"
if not defined UMA_LAUNCH_CONDA exit /b 0
for %%C in ("%UMA_LAUNCH_CONDA%") do (
    for %%A in ("%%~dpC..\condabin\conda.bat" "%%~dpCcondabin\conda.bat") do if not defined UMA_LAUNCH_ACTIVATE if exist "%%~A" set "UMA_LAUNCH_ACTIVATE=%%~fA"
)
exit /b 0

:failed
if not defined UMA_LAUNCH_CHECK pause
exit /b 1
