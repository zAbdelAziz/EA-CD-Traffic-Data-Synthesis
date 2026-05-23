from pathlib import Path

def count_lines_py(root: str) -> tuple[int, int]:
    root_path = Path(root)
    total_lines = 0
    file_count = 0

    for path in root_path.rglob("*.py"):
        if path.is_file():
            file_count += 1
            # count newline characters efficiently
            with path.open("rb") as f:
                total_lines += f.read().count(b"\n")
            # if file doesn't end with '\n', the above undercounts by 1
            with path.open("rb") as f:
                data = f.read()
                if data and not data.endswith(b"\n"):
                    total_lines += 1

    return total_lines, file_count

if __name__ == "__main__":
    total, files = count_lines_py(".")
    print(f"Python files: {files}")
    print(f"Total lines:  {total}")