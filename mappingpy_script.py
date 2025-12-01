import os
import json
from maplines import lhdiff

DATASET_DIR = "evaluate_set"
OUTPUT_DIR = "evaluate_set"

# Default LHDiff parameters
K = 15
Thigh = 0.85
Tmid = 0.85
WINDOW = 4
MAXSPAN = 3

def format_result(result: dict) -> str:

    # Prepare a new dictionary to save the content to be written into JSON
    output = {}

   
    output["old_file_total_lines"] = result["old_file_total_lines"]
    output["new_file_total_lines"] = result["new_file_total_lines"]

    # "mapping" is a list, each item is a dictionary
    mapping_list = []
    for item in result["mapping"]:
        one_mapping = {}
        one_mapping["old_line"] = item["old_line"]
        one_mapping["new_lines"] = item["new_lines"]
        one_mapping["type"] = item["type"]
        one_mapping["note"] = item["note"]
        mapping_list.append(one_mapping)

    output["mapping"] = mapping_list

    # Convert the entire dictionary into a formatted JSON string
    json_text = json.dumps(
        output,
        ensure_ascii=False,  # Keep non-ASCII characters
        indent=2             # looks better
    )

    return json_text



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