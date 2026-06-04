from pathlib import Path
import shutil
from datetime import datetime

# ============================
# CHANGE THIS TO YOUR PROJECT PATH
# ============================
PROJECT_ROOT = Path(r"C:\Users\ASUS\PycharmProjects\PythonProject1")

# Existing Outputs folder
SOURCE_OUTPUTS = PROJECT_ROOT / "Outputs"

# New folder name
DEST_FOLDER = PROJECT_ROOT / "Final_Report_Outputs"

# ZIP name
ZIP_NAME = PROJECT_ROOT / "Final_Report_Outputs_ZIP"


def copy_outputs():
    if not SOURCE_OUTPUTS.exists():
        raise FileNotFoundError(f"Outputs folder not found: {SOURCE_OUTPUTS}")

    # If old final folder exists, rename it with timestamp
    if DEST_FOLDER.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = PROJECT_ROOT / f"Final_Report_Outputs_backup_{timestamp}"
        DEST_FOLDER.rename(backup_folder)
        print(f"Old Final_Report_Outputs renamed to: {backup_folder}")

    # Copy full Outputs folder structure
    shutil.copytree(SOURCE_OUTPUTS, DEST_FOLDER)

    print("\n✅ Outputs copied successfully!")
    print(f"From: {SOURCE_OUTPUTS}")
    print(f"To  : {DEST_FOLDER}")

    # Create ZIP file
    zip_path = shutil.make_archive(str(ZIP_NAME), "zip", DEST_FOLDER)

    print("\n✅ ZIP file created successfully!")
    print(f"ZIP: {zip_path}")

    # Create summary file
    summary_file = DEST_FOLDER / "README_OUTPUTS_SUMMARY.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("FINAL REPORT OUTPUTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source folder: {SOURCE_OUTPUTS}\n")
        f.write(f"Copied folder: {DEST_FOLDER}\n")
        f.write(f"ZIP file: {zip_path}\n\n")
        f.write("Folder structure copied:\n\n")

        for path in DEST_FOLDER.rglob("*"):
            relative_path = path.relative_to(DEST_FOLDER)
            if path.is_dir():
                f.write(f"[FOLDER] {relative_path}\n")
            else:
                f.write(f"        {relative_path}\n")

    print("\n✅ Summary file created!")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    copy_outputs()