import argparse
import re, json, argparse, difflib, math, hashlib

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

        