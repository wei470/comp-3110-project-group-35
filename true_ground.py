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


def process_pair(idx):
    old_f = f"evaluate_set/old_{idx}.txt"
    new_f = f"evaluate_set/new_{idx}.txt"

    old_lines = load_file(old_f)
    new_lines = load_file(new_f)

    gt_lines = generate_gt(old_lines, new_lines)

    out_f = f"evaluate_set/pair{idx}.txt"
    with open(out_f, "w", encoding="utf-8") as f:
        f.write("\n".join(gt_lines))

    print("Generated:", out_f)

def main():
    for i in range(1, 25):
        process_pair(i)

main()
