import os
import sys

BASE_DIR = "/gws/ssde/j25a/ai4er/users/pn341/climate-rl"


def validate_args():
    if len(sys.argv) != 2:
        print("Usage: python rename_infx10.py <target_subdir>")
        print("Example: python rename_infx10.py records")
        sys.exit(1)

    subdir = sys.argv[1]
    target_root = os.path.join(BASE_DIR, subdir)

    if not os.path.isdir(target_root):
        print(f"Error: {target_root} is not a valid directory.")
        sys.exit(1)

    return target_root


def rename_folders(root_dir: str) -> None:
    for run_folder in os.listdir(root_dir):
        if not run_folder.startswith("infx10_"):
            continue

        run_path = os.path.join(root_dir, run_folder)
        if not os.path.isdir(run_path):
            continue

        for child in os.listdir(run_path):
            old_path = os.path.join(run_path, child)
            if not os.path.isdir(old_path):
                continue

            chunks = child.split("__")

            # Only rename if there are ≥ 5 chunks (i.e., 4+ double-underscores)
            if len(chunks) < 5:
                continue

            # Remove 3rd chunk (index 2)
            new_chunks = chunks[:2] + chunks[3:]
            new_name = "__".join(new_chunks)

            if new_name != child:
                new_path = os.path.join(run_path, new_name)
                print(f"Renaming: {old_path} → {new_path}")
                os.rename(old_path, new_path)


def main():
    target_root = validate_args()
    rename_folders(target_root)


if __name__ == "__main__":
    main()
