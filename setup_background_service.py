"""Set up Mindflow keystroke background service to start at user logon on Windows."""

import os
import subprocess
from pathlib import Path


def _write_runner_bat(mindflow_dir: Path, python_exe: Path, service_script: Path) -> Path:
    runner = mindflow_dir / "run_keystroke_service.bat"
    content = f"""@echo off
cd /d "{mindflow_dir}"
"{python_exe}" "{service_script}"
"""
    runner.write_text(content, encoding="utf-8")
    return runner


def _try_create_schtask(task_name: str, runner_bat: Path) -> tuple[bool, str]:
    delete_cmd = f'schtasks /delete /tn "{task_name}" /f'
    subprocess.run(delete_cmd, shell=True, capture_output=True, text=True)

    create_cmd = (
        f'schtasks /create /tn "{task_name}" '
        f'/tr "\"{runner_bat}\"" '
        f'/sc onlogon '
        f'/rl LIMITED '
        f'/f'
    )
    result = subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
    ok = result.returncode == 0
    msg = (result.stdout or result.stderr or "").strip()
    return ok, msg


def _create_startup_shortcut(runner_bat: Path) -> tuple[bool, str]:
    startup = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    if not startup.exists():
        return False, f"Startup folder not found: {startup}"

    shortcut_path = startup / "MindflowKeystrokeService.lnk"
    shortcut_ps = str(shortcut_path).replace("'", "''")
    target_ps = str(runner_bat).replace("'", "''")
    workdir_ps = str(runner_bat.parent).replace("'", "''")

    # Use PowerShell COM automation to create shortcut.
    ps = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut('{shortcut_ps}'); "
        f"$s.TargetPath = '{target_ps}'; "
        f"$s.WorkingDirectory = '{workdir_ps}'; "
        "$s.Save();"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    ok = result.returncode == 0 and shortcut_path.exists()
    msg = (result.stdout or result.stderr or "").strip()
    if ok and not msg:
        msg = f"Created startup shortcut: {shortcut_path}"
    return ok, msg


def setup_background_service() -> None:
    mindflow_dir = Path(__file__).resolve().parent
    python_exe = mindflow_dir / ".venv" / "Scripts" / "python.exe"
    service_script = mindflow_dir / "keystroke_background_service.py"
    task_name = "MindflowKeystrokeService"

    print("\n" + "=" * 60)
    print("MINDFLOW KEYSTROKE BACKGROUND SERVICE SETUP")
    print("=" * 60)

    if not python_exe.exists():
        print(f"Python not found: {python_exe}")
        return
    if not service_script.exists():
        print(f"Service script not found: {service_script}")
        return

    print("\n[Step 1] Creating runner batch file...")
    runner_bat = _write_runner_bat(mindflow_dir, python_exe, service_script)
    print(f"Created: {runner_bat}")

    print("\n[Step 2] Configuring auto-start...")
    task_ok, task_msg = _try_create_schtask(task_name, runner_bat)
    if task_ok:
        print(f"Task Scheduler configured: {task_name}")
    else:
        print("Task Scheduler creation failed, trying Startup folder fallback...")
        if task_msg:
            print(task_msg)

        startup_ok, startup_msg = _create_startup_shortcut(runner_bat)
        if startup_ok:
            print("Startup folder fallback configured.")
            print(startup_msg)
        else:
            print("Could not configure Startup folder fallback automatically.")
            if startup_msg:
                print(startup_msg)
            print("Manual fallback: put run_keystroke_service.bat into your Startup folder.")

    print("\n[Step 3] Quick service startup check...")
    result = subprocess.run(
        [str(python_exe), "-c", "import keystroke_background_service as s; print('import_ok')"],
        cwd=str(mindflow_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("Service import check passed.")
    else:
        print("Service import check failed:")
        print((result.stderr or result.stdout).strip())

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("Log file: data/keystroke/background_service.log")
    print("Baseline file: data/keystroke/baseline_auto.json")
    print("Run now: .\\run_keystroke_service.bat")


if __name__ == "__main__":
    setup_background_service()
