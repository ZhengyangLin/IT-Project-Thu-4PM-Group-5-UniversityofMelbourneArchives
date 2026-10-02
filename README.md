# IT-Project-group5
melbourne archives

## One-click startup (Windows)

Double-click **start.bat** in the project root. It checks the local environment,
then opens two **visible CMD** windows for the backend and frontend, with
live logs in each window. The launcher is a plain batch file and does not invoke
PowerShell. Once both services report that they are ready, open
<http://127.0.0.1:5173>. Stop each service with **Ctrl+C** in its window, or close
both service windows. Errors remain visible in the service windows.
The service windows run Python and Node directly, so Ctrl+C stops the service
without a `Terminate batch job (Y/N)?` confirmation.

Install the backend dependencies from `requirements.txt` in your Python/OCR
environment first, and run `yarn install --frozen-lockfile` inside `front-end`.
Node.js must satisfy the version range in `front-end/package.json`. The launcher
uses the installed Vite directly and does not install or update packages.

Python selection: `-PythonExe` / `UMA_PYTHON`, project `.venv` / `venv`, active
virtualenv / Conda environment, `uma` under `%USERPROFILE%\miniconda3` or
`%USERPROFILE%\anaconda3`, then `python` / `py` on PATH. Conda is activated in the
launcher before opening the service windows, which inherit its settings and DLL paths.
For an environment elsewhere, specify its Python executable:

```bat
start.bat -PythonExe "D:\my environments\uma\python.exe"
```

For future double-click launches, set the `UMA_PYTHON` Windows user environment
variable to that executable path. To check prerequisites without starting either
service:

```bat
start.bat -Check
```

The backend uses port **8000** (required by the frontend's API URLs); the frontend
uses **5173**. If either port is occupied, the launcher reports it without stopping
existing processes. Missing Python modules must be installed in the Python
environment printed by the launcher. Backend API docs: <http://127.0.0.1:8000/docs>.
