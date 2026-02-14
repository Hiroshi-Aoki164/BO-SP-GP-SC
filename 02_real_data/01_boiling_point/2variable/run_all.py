"""Main script to run all scripts sequentially."""
import subprocess
import sys
from datetime import datetime

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting {script_name}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout:
            print(result.stdout)

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {script_name} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] An error occurred while running {script_name}")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred while running {script_name}: {e}")
        return False

def main():
    """Main function that runs all scripts sequentially."""
    print("="*60)
    print("Starting batch execution of all scripts")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    scripts = [
        "MLR.py",
        "BO_GPR.py",
        "BO_SP_GPR.py",
        "BO_SP_GPR_SC.py"
    ]

    results = {}

    for script in scripts:
        success = run_script(script)
        results[script] = "Success" if success else "Failure"

    print("\n" + "="*60)
    print("Execution summary")
    print("="*60)
    for script, status in results.items():
        print(f"{script}: {status}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if "Failure" in results.values():
        sys.exit(1)

if __name__ == "__main__":
    main()
