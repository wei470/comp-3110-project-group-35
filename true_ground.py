import os
os.chdir(os.path.dirname(__file__))   # 保证从 evaluate_set 运行

def load_file(path):
    lines = []

    # Use with to open the file, so it will automatically close after use
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove newlines at the end of each line
            clean_line = line.rstrip("\n")
            lines.append(clean_line)

    return lines


def generate_gt(old_lines, new_lines):
    mapping = []

    #old_index starts from 1, keeping the line number consistent with the line number in the text file
    for old_index, old_line in enumerate(old_lines, start=1):
        match_positions = []

        #Find all lines in new_lines that are the same as old_line
        for new_index, new_line in enumerate(new_lines, start=1):
            if new_line == old_line:
                match_positions.append(new_index)

        #Generate different strings based on number of matches
        if len(match_positions) == 0:
            line_text = f"{old_index} -> (deleted)"
        elif len(match_positions) == 1:
            line_text = f"{old_index} -> {match_positions[0]}"
        else:
            #Spell multiple line numbers into the form "a, b, c"
            joined = ", ".join(str(pos) for pos in match_positions)
            line_text = f"{old_index} -> {joined}  (duplicated)"

        mapping.append(line_text)

    return mapping


def process_pair(index):  
"""
    Process a pair of old_index.txt and new_index.txt files,
    Call generate_gt to generate the corresponding ground truth,
    Finally write evaluate_set/pair{index}.txt.
    """

    # Spell out the paths of old and new files
    old_path = os.path.join("evaluate_set", f"old_{index}.txt")
    new_path = os.path.join("evaluate_set", f"new_{index}.txt")

    # Read the old file and the new file in line by line.
    old_lines = load_file(old_path)
    new_lines = load_file(new_path)

    # Call generate_gt to generate the mapping result (each item is a string of "line number -> ...")
    gt_lines = generate_gt(old_lines, new_lines)

    # Output file path: pair{index}.txt
    out_path = os.path.join("evaluate_set", f"pair{index}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for line in gt_lines:
            f.write(line + "\n")

    print(f"Generated ground truth file: {out_path}")


def main():
    for i in range(1, 25):
        process_pair(i)

main()
