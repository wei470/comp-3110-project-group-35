import json

def load_gt(path):
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            old, right = line.strip().split(" -> ")
            old = int(old)
            if right == "(deleted)":
                gt[old] = []
            else:
                nums = [int(x) for x in right.replace("(duplicated)", "").split(",")]
                gt[old] = nums
    return gt

def load_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mp = {e["old_line"]: e["new_lines"] for e in data["mapping"]}
    return mp

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
    print("pair, correct%, correct, change, spurious, eliminate")

    for i in range(1, 25):
        gt = load_gt(f"evaluate_set/pair{i}.txt")
        mp = load_mapping(f"evaluate_set/mapping_{i}.json")

        correct, change, spurious, eliminate = compare(gt, mp)

        total = len(gt)
        correct_rate = correct / total * 100

        print(f"{i}, {correct_rate:.2f}%, {correct}, {change}, {spurious}, {eliminate}")

main()
