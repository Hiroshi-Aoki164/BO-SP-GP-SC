import subprocess
import sys
import os

def run_script(folder_path, script_name="run_all.py"):
    script_path = os.path.join(folder_path, script_name)

    if not os.path.exists(script_path):
        print(f"エラー: {script_path} が見つかりません")
        return False

    print(f"\n{'='*60}")
    print(f"{folder_path} 内の {script_name} を実行中...")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=folder_path,
            check=True,
            capture_output=False
        )
        print(f"\n{folder_path} の実行が完了しました")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nエラー: {folder_path} の実行中にエラーが発生しました")
        print(f"エラーコード: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n予期しないエラー: {e}")
        return False

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    folders = [
        os.path.join(base_dir, "9variable"),
        os.path.join(base_dir, "2variable")
    ]

    print("run_all.py の順次実行を開始します\n")

    results = []
    for folder in folders:
        success = run_script(folder)
        results.append((folder, success))

    print(f"\n{'='*60}")
    print("実行結果サマリー")
    print(f"{'='*60}")
    for folder, success in results:
        status = "成功" if success else "失敗"
        print(f"{os.path.basename(folder)}: {status}")

    if all(success for _, success in results):
        print("\nすべてのスクリプトが正常に完了しました")
    else:
        print("\n一部のスクリプトが失敗しました")
        sys.exit(1)
