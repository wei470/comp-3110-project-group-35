import json

def load_gt(path):
    """
   Read the ground truth file (pairX.txt),
    Returns a dictionary: key is the old line number, value is the corresponding new line number list.
    For example:
      1 -> 2           => {1: [2]}
      2 -> (deleted)   => {2: []}
      3 -> 4, 5        => {3: [4, 5]}
    """
    gt = {}

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                # Blank lines are skipped directly (although they are generally not included in normal input)
                continue

            left_and_right = line.split(" -> ")
            if len(left_and_right) != 2:
               # If the format is wrong, skip it first so that the program will not crash directly.
                # It can also be changed to raise Exception
                continue

            old_str, right_str = left_and_right
            old_line_no = int(old_str)

            if right_str == "(deleted)":
               # Marked for deletion, corresponding to the empty list
                gt[old_line_no] = []
            else:
                # Remove possible "(duplicated)" tags
                cleaned = right_str.replace("(duplicated)", "")

                # Split with commas
                pieces = cleaned.split(",")

                new_line_list = []
                for piece in pieces:
                    piece = piece.strip()
                    if piece:  # Prevent empty strings from appearing

                        new_line_list.append(int(piece))

                gt[old_line_no] = new_line_list

    return gt


def load_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}

   # data["mapping"] is a list, each item is a dictionary
    for item in data["mapping"]:
        old_line = item["old_line"]
        new_lines = item["new_lines"]

       # Assume the input is legal
        result[old_line] = new_lines

    return result


def compare(gt, mp):
    correct = 0
    change = 0
    spurious = 0
    eliminate = 0

    for old_line, gt_new in gt.items():
        out_new = mp.get(old_line, [])

        if out_new == gt_new:
            correct += 1
        else:
            if out_new and not gt_new:
                # 本来应该 deleted，你输出了映射
                spurious += 1
            elif not out_new and gt_new:
                # 本来应该有映射，你没输出
                eliminate += 1
            else:
                change += 1

    return correct, change, spurious, eliminate

def main():
    """
    Traverse samples 1~24 in order:
    - Read ground truth (pairX.txt)
    - Read the mapping_X.json we generated
    - Call compare_mapping to count four situations
    - Print one line of simple statistical results
    """
    print("pair, correct%, correct, change, spurious, eliminate")

    for pair_index in range(1, 25):
        gt_path = f"evaluate_set/pair{pair_index}.txt"
        mapping_path = f"evaluate_set/mapping_{pair_index}.json"

        #ground truth and output results are both read into dictionaries
        gt_dict = load_gt(gt_path)
        mapping_dict = load_mapping(mapping_path)

        correct, change, spurious, eliminate = compare_mapping(gt_dict, mapping_dict)

        total_lines = len(gt_dict)

       # Prevent division by zero when total_lines is 0
        if total_lines == 0:
            correct_rate = 0.0
        else:
            correct_rate = correct / total_lines * 100.0

        print(
            f"{pair_index}, "
            f"{correct_rate:.2f}%, "
            f"{correct}, "
            f"{change}, "
            f"{spurious}, "
            f"{eliminate}"
        )


if __name__ == "__main__":
    main()

