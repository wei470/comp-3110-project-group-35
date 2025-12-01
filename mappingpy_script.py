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

    # If the output directory does not exist, create one first
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # collect all old_x.txt file names into a list
    old_files = []
    for file_name in os.listdir(DATASET_DIR):
        if file_name.startswith("old_") and file_name.endswith(".txt"):
            old_files.append(file_name)

    # Sort it and ensure that the processing order is old_1, old_2, ...
    old_files.sort()

    # Process each old_x.txt in turn
    for old_file in old_files:
        # "old_3.txt" takes the middle number part "3"
        index = old_file[4:-4]

        # Get the full paths of old_x.txt and new_x.txt
        old_path = os.path.join(DATASET_DIR, f"old_{index}.txt")
        new_path = os.path.join(DATASET_DIR, f"new_{index}.txt")

        # Use readlines to retain line breaks at the end of the line and give them to lhdiff for use
        with open(old_path, "r", encoding="utf-8", errors="ignore") as f:
            old_lines = f.readlines()

        with open(new_path, "r", encoding="utf-8", errors="ignore") as f:
            new_lines = f.readlines()

        # Call lhdiff to get the mapping result
        result = lhdiff(
            old_lines,
            new_lines,
            K,
            Thigh,
            Tmid,
            WINDOW,
            MAXSPAN
        )

        # Convert to JSON text
        json_text = format_result(result)

        # Output file name mapping_x.json
        out_path = os.path.join(OUTPUT_DIR, f"mapping_{index}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_text)

        print(f"script run completed for pair {index}")



if __name__ == "__main__":
    run_all()