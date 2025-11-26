import os
import json
from maplines import lhdiff

DATASET_DIR = "evaluate_set"
OUTPUT_DIR = "results"

# Default LHDiff parameters
K = 10
Thigh = 0.85
Tmid = 0.65
WINDOW = 4
MAXSPAN = 2

def format_result(result: dict) -> str:

    # header
    header = (
        '{\n'
        f'  "old_file_total_lines": {result["old_file_total_lines"]},\n'
        f'  "new_file_total_lines": {result["new_file_total_lines"]},\n'
        '  "mapping": [\n'
    )

    lines = []
    for e in result["mapping"]:
        line = (
            '    { '
            f'"old_line": {e["old_line"]}, '
            f'"new_lines": {json.dumps(e["new_lines"], ensure_ascii=False)}, '
            f'"type": {json.dumps(e["type"], ensure_ascii=False)}, '
            f'"note": {json.dumps(e["note"], ensure_ascii=False)} '
            '}'
        )
        lines.append(line)

    body = ",\n".join(lines)
    tail = '\n  ]\n}\n'

    return header + body + tail


def run_all():
    # ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # find old_x.txt
    files = sorted(
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("old_") and f.endswith(".txt")
    )

    for old_file in files:
        # old_3.txt → "3"
        index = old_file[4:-4] 

        # old_x.txt and new_x.txt
        old_path = os.path.join(DATASET_DIR, f"old_{index}.txt")
        new_path = os.path.join(DATASET_DIR, f"new_{index}.txt")

        with open(old_path, encoding="utf-8", errors="ignore") as f:
            old_lines = f.readlines()

        with open(new_path, encoding="utf-8", errors="ignore") as f:
            new_lines = f.readlines()

        result = lhdiff(old_lines, new_lines, K, Thigh, Tmid, WINDOW, MAXSPAN)

        json_text = format_result(result)

        # saved to mapping_x.json
        out_path = os.path.join(OUTPUT_DIR, f"mapping_{index}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_text)

        print(f"script run completed")


if __name__ == "__main__":
    run_all()
