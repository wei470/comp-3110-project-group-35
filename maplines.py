import argparse
import re, json, argparse, difflib, math, hashlib

# remove extra space
def normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

#lower case all letter
def token_iter(s: str):
    return re.findall(r"[A-Za-z0-9_]+", s.lower())

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

        