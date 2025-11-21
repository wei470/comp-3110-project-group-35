import argparse
import re, json, argparse, difflib, math, hashlib
from collections import Counter

# remove extra space
def normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

#lower case all letter
def token_iter(s: str):
    return re.findall(r"[A-Za-z0-9_]+", s.lower())

# convert string into 64-bit SimHash
def simhash64(s: str) -> int:
    v = [0]*64
    for tok in token_iter(s):
        h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum((1 << i) for i, val in enumerate(v) if val >= 0)

# compare 64-bit SimHash of 2 strings
def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# how many steps to turn string a to b using edit/delete/subsitute letter
# using dynamic programming
# idea: Levenshtein distance
def edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb+1):
            prev, dp[j] = dp[j], min(prev + (a[i-1] != b[j-1]), dp[j] + 1, dp[j-1] + 1)
    return dp[lb]

# similarly of two string
# 0 -> two string are not the same at all
# 1 -> two string are identital or one of them or both are empty string
def edit_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    elif not a and not b:
        return 1.0
    else:
        return 1 - edit_distance(a, b) / max(1, max(len(a), len(b)))

# Compare the bag-of-words similarity between the two passages in the "near context".
# dot = for each element c1' in c1 and each element c2', c1'*c2' iff c1=c2
# ie: 
#   c1 = Counter(["def", "add", "x", "y", "return", "x", "y"])
#   c2 = Counter(["def", "sum", "a", "b", "return", "a", "b"])
# same element in both c1 and c2: "def" , "return", they all showed once in c1 and c2
# dot = 1+1 = 2
# n1 = sqrt (each element in c1 * # of occurence of this element)

# ie
# c1 = {"def":1, "add":1, "x":2, "y":2, "return":1}
# n1 = sqrt(1^2 + 1^2 + 2^2 + 2^2 + 1^2) = sqrt(11)
# c2 = {"def":1, "sum":1, "a":2, "b":2, "return":1}
# n1 = sqrt(11)

# return "cosine" =  dot / (n1 * n2)
# 
# 0 -> two string are not the same at all
# 1 -> two string are identital or one of them or both are empty string

def cosine_sim(c1: Counter, c2: Counter) -> float:
    dot = sum(c1[k] * c2[k] for k in c1 if k in c2)
    n1 = math.sqrt(sum(v*v for v in c1.values()))
    n2 = math.sqrt(sum(v*v for v in c2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0

# merge i+windows lines and i-windows lines into one line
# ie
#lines = 
#    "int a = 0;",
#    "a += 1;",
#    "return a;",
#    "}",
#    "// end"
#

# bow_context(lines, i=2, window=2)
# i = 2 -> lines 2 -> int a = 0;
# sinces windows = 2, merge i = [0 ~ 4 into one line]
# return "int a = 0; a +=1; } // end"

def bow_context(lines, i, window):
    L, R = max(0, i - window), min(len(lines)-1, i + window)
    return " ".join(lines[L:i] + lines[i+1:R+1])

# main process
def lhdiff(old_lines, new_lines):
    
    # step 1: normalize all input files
    # ppt 20
    old_norm = [normalize_line(x) for x in old_lines]
    new_norm = [normalize_line(x) for x in new_lines]

    # step 2: Detect Unchanged Lines
    # pg 26-27

    # anchors: record # line of old_line → # line of [new_line] where old_line = new_line
    # ie:
    # old_norm = [
    # "int a = 1;",      # 0
    # "int b = 2;",      # 1
    # "return a + b;",   # 2
    # ]

    # new_norm = [
    # "int a = 1;",      # 0
    # "int b = 3;",      # 1  （changed）
    # "return a + b;",   # 2

    # sm =
    # ("equal",   0, 1,   0, 1),   # old[0] == new[0]
    # ("replace", 1, 2,   1, 2),   # old[1] != new[1]
    # ("equal",   2, 3,   2, 3),   # old[2] == new[2]

    # anchors =
    # 0: [0],   # old[1] = new[0]
    # 2: [2],    # old[2] = new[2]

    sm = difflib.SequenceMatcher(a=old_norm, b=new_norm, autojunk=False)
    anchors = {i: [j] for tag, i1, i2, j1, j2 in sm.get_opcodes()
               if tag == "equal" for i, j in zip(range(i1, i2), range(j1, j2))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("old_file")
    ap.add_argument("new_file")
    ap.add_argument("-o", "--output", default="mapping.json")
    args = ap.parse_args()
    
    with open(args.old_file, encoding="utf-8", errors="ignore") as f:
        old_lines = f.readlines()
    with open(args.new_file, encoding="utf-8", errors="ignore") as f:
        new_lines = f.readlines()

        