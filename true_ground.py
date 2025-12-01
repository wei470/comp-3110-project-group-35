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
    for oi, line in enumerate(old_lines):
        matches = [nj + 1 for nj, nl in enumerate(new_lines) if nl == line]

        if not matches:
            mapping.append(f"{oi+1} -> (deleted)")
        elif len(matches) == 1:
            mapping.append(f"{oi+1} -> {matches[0]}")
        else:
            mapping.append(f"{oi+1} -> {', '.join(map(str, matches))}  (duplicated)")
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
