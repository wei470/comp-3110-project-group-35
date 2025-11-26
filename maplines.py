import re
import json
import argparse
import difflib
import math
import hashlib
from collections import Counter

# Global debugging switch: debugging information is not printed by default
global DEBUG
DEBUG = False

def debug_print(*args, **kwargs):
   
    if DEBUG:
        print(*args, **kwargs)

# remove extra space
def normalize_line(s: str) -> str:
    return s.rstrip("\n")

#lower case all letter
def token_iter(s: str):
    return re.findall(r"[A-Za-z0-9_]+", s.lower())

# convert string into 64-bit SimHash
def simhash64(s: str) -> int:
    # Record the "number of votes" for each bit
    bit_votes = [0] * 64

    # First count all the tokens to facilitate printing of debugging information.
    tokens = list(token_iter(s))
    debug_print(f"[simhash64] input string: {s!r}")
    debug_print(f"[simhash64] tokens: {tokens}")

    for tok in tokens:
        # Calculate 64-bit hash value for each token
        h_hex = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest()
        h_int = int(h_hex, 16)

        # Print the hash of each token-select hex
        debug_print(f"  [simhash64] token={tok!r}, hash_hex={h_hex}, hash_int={h_int:#018x}")

        # Traverse 64 bits and convert 1/0 to +1/-1
        for bit_index in range(64):
            if (h_int >> bit_index) & 1:
                bit_votes[bit_index] += 1
            else:
                bit_votes[bit_index] -= 1

    # Construct the final 64-bit integer based on the "votes" of each bit
    result_hash = 0
    for bit_index, vote in enumerate(bit_votes):
        if vote >= 0:
            result_hash |= (1 << bit_index)

    debug_print(f"[simhash64] result hash for string {s!r}: {result_hash:#018x}")
    return result_hash


# compare 64-bit SimHash of 2 strings
# smallest the return, the more similary they are
# int a = 10  -> 10100110101010
# int b = 10  -> 10100110111010
# hamming(a,b) = 1~2

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
def lhdiff(old_lines, new_lines, k, Thigh, Tmid, window, maxspan=2):
  
    # Standardization of lines (remove extra spaces, etc.)
    old_norm = [normalize_line(x) for x in old_lines]
    new_norm = [normalize_line(x) for x in new_lines]

    debug_print("===== Step 1: normalized lines =====")
    for idx, line in enumerate(old_norm):
        debug_print(f"[old_norm] line {idx}: {line!r}")
    for idx, line in enumerate(new_norm):
        debug_print(f"[new_norm] line {idx}: {line!r}")

    #Use difflib to find exactly the same line as an "anchor"
    sm = difflib.SequenceMatcher(a=old_norm, b=new_norm, autojunk=False)
    anchors = {}

    anchors = {}
    for i, line_old in enumerate(old_norm):
        for j, line_new in enumerate(new_norm):
            if line_old == line_new:
                anchors[i] = [j]
                break  # one anchor

    debug_print("===== Step 2: anchors (exact matches) =====")
    debug_print(anchors)

   # Preprocessing context, SimHash and bag-of-words
    #Context: Center the current line, take the front and rear window lines, and concatenate them into a large string
    old_ctx = []
    for i in range(len(old_norm)):
        ctx_str = bow_context(old_norm, i, window)
        old_ctx.append(ctx_str)

    new_ctx = []
    for i in range(len(new_norm)):
        ctx_str = bow_context(new_norm, i, window)
        new_ctx.append(ctx_str)

    debug_print("===== Step 3: context strings around each line =====")
    for idx, ctx in enumerate(old_ctx):
        debug_print(f"[old_ctx] line {idx}: {ctx!r}")
    for idx, ctx in enumerate(new_ctx):
        debug_print(f"[new_ctx] line {idx}: {ctx!r}")

    # Calculate the simhash of each row itself and the context
    old_sh = []
    old_shctx = []
    for idx, line in enumerate(old_norm):
        h_line = simhash64(line)
        h_ctx = simhash64(old_ctx[idx])
        old_sh.append(h_line)
        old_shctx.append(h_ctx)
        debug_print(f"[old_hash] line {idx}: line_hash={h_line:#018x}, ctx_hash={h_ctx:#018x}")

    new_sh = []
    new_shctx = []
    for idx, line in enumerate(new_norm):
        h_line = simhash64(line)
        h_ctx = simhash64(new_ctx[idx])
        new_sh.append(h_line)
        new_shctx.append(h_ctx)
        debug_print(f"[new_hash] line {idx}: line_hash={h_line:#018x}, ctx_hash={h_ctx:#018x}")

    # Prepare bag-of-words Counter for the context of new
    new_bowctx = []
    for ctx in new_ctx:
        c = Counter(token_iter(ctx))
        new_bowctx.append(c)

    debug_print("===== Step 3: new context bag-of-words =====")
    for idx, c in enumerate(new_bowctx):
        debug_print(f"[new_bowctx] line {idx}: {c}")

    # Helper function
    def span_text(lo: int, hi: int) -> str:
        return " ".join(new_norm[lo:hi + 1])
    def span_ctx_counter(lo: int, hi: int) -> Counter:
       
        combined = Counter()
        for line_index in range(lo, hi + 1):
            combined.update(new_bowctx[line_index])
        return combined

    mapping = {}
    N = len(new_norm)

    # For each line old[i], find the most suitable paragraph in new
    for oi, old_line in enumerate(old_norm):
        if oi in anchors:
            mapping[oi] = anchors[oi]
            debug_print(f"[mapping] old line {oi} is exact match -> new lines {anchors[oi]}")
            continue

        # 1) First use simhash distance to roughly filter out the top-k candidate new lines
        candidate_scores = []
        for nj in range(N):
            dist = hamming(old_sh[oi], new_sh[nj])
            candidate_scores.append((nj, dist))

        candidate_scores.sort(key=lambda x: x[1])
        base_indices = [pair[0] for pair in candidate_scores[:k]]
        debug_print(f"[candidates] old line {oi}: base_indices={base_indices}")

        # 2) Using these new lines as centers, construct various possible spans (lengths from 1 to maxspan)
        cand_spans = set()
        for j in base_indices:
            for span_len in range(1, maxspan + 1):
                # Right
                lo1 = j
                hi1 = min(N - 1, j + span_len - 1)
                cand_spans.add((lo1, hi1))

                # Left
                hi2 = j
                lo2 = max(0, j - (span_len - 1))
                cand_spans.add((lo2, hi2))

        # Filter out out-of-bounds spans and sort them
        valid_spans = []
        for (lo, hi) in cand_spans:
            if 0 <= lo <= hi < N:
                valid_spans.append((lo, hi))
        valid_spans.sort()

        debug_print(f"[cand_spans] old line {oi}: {valid_spans}")

        # 3) Calculate the final score (combined similarity of content + context) for each candidate span
        src_ctx_counter = Counter(token_iter(old_ctx[oi]))
        best_score = -1.0
        best_span = None

        for (lo, hi) in valid_spans:
            new_text = span_text(lo, hi)
            content_s = edit_sim(old_line, new_text)
            ctx_s = cosine_sim(src_ctx_counter, span_ctx_counter(lo, hi))
            score = 0.6 * content_s + 0.4 * ctx_s

            debug_print(
                f"[score] old line {oi}, span ({lo},{hi}): "
                f"content_s={content_s:.4f}, ctx_s={ctx_s:.4f}, score={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_span = (lo, hi)

        # 4) If the best span score reaches the threshold, record it in mapping
        if best_span is not None and best_score >= Tmid:
            lo, hi = best_span
            mapping[oi] = list(range(lo, hi + 1))
            debug_print(
                f"[mapping] old line {oi} best_span={best_span}, "
                f"score={best_score:.4f}, mapped_to={mapping[oi]}"
            )
        else:
            debug_print(
                f"[mapping] old line {oi}: no suitable span found "
                f"(best_score={best_score:.4f}, Tmid={Tmid})"
            )

    #Generate the final result list and detect "multiple lines old -> single line new (merge)"
    # Construct a reverse mapping: new rows -> which old rows are mapped
    reverse_map = {}
    for old_idx, new_list in mapping.items():
        for nj in new_list:
            reverse_map.setdefault(nj, []).append(old_idx)

    debug_print("===== Step 5: reverse_map (new line -> list of old lines) =====")
    debug_print(reverse_map)

    result = []
    old_total = len(old_norm)
    new_total = len(new_norm)

    for oi in range(old_total):
        entry = {
            "old_line": oi + 1,
            "new_lines": [],
            "type": "deleted",
            "note": ""
        }

        if oi in mapping:
            njs = sorted(mapping[oi])
            entry["new_lines"] = [j + 1 for j in njs]

           # Put the corresponding new lines into one line, and then compare it with old_norm[oi]
            joined_new = " ".join(new_norm[j] for j in njs)

            if old_norm[oi] == joined_new:
                entry["type"] = "unchanged"
            else:
                entry["type"] = "modified"

            # One line old is split into multiple lines new (split)
            if len(njs) >= 2:
                entry["note"] = f"split into {len(njs)} lines"
            # Handle the situation of "multiple lines old -> single line new" (merge)
            elif len(njs) == 1:
                nj = njs[0]
                old_list_for_this_new = reverse_map.get(nj, [])
                if len(old_list_for_this_new) >= 2:
                    # Such as "merged with 2 other lines"
                    entry["note"] = f"merged with {len(old_list_for_this_new) - 1} other lines"

        result.append(entry)

    return {
        "old_file_total_lines": old_total,
        "new_file_total_lines": new_total,
        "mapping": result
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # name of two files to compare
    ap.add_argument("file_1", nargs="?", default="old.txt")
    ap.add_argument("file_2", nargs="?", default="new.txt")

    # deafult k = 10, thigh = 0.85, Tmid = 0.65, window = 4, maxspan = 2
    ap.add_argument("-o", "--output", default="mapping.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--Thigh", type=float, default=0.85)
    ap.add_argument("--Tmid", type=float, default=0.65)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--maxspan", type=int, default=2)

    # Switch
    ap.add_argument(
        "--debug",
        action="store_true",
        help="print detailed debug information to the terminal"
    )

    args = ap.parse_args()

    # Set global DEBUG variable based on command line parameters
    DEBUG = args.debug


    with open(args.file_1, encoding="utf-8", errors="ignore") as f:
        old_lines = f.readlines()
    with open(args.file_2, encoding="utf-8", errors="ignore") as f:
        new_lines = f.readlines()

    result = lhdiff(old_lines, new_lines, args.k, args.Thigh, args.Tmid, args.window, args.maxspan)

    
    # header print old and new file total lines
    header = (
        '{\n'
        f'  "old_file_total_lines": {result["old_file_total_lines"]},\n'
        f'  "new_file_total_lines": {result["new_file_total_lines"]},\n'
        '  "mapping": [\n'
    )

    # print old -> new mapping into lines[]
    # includes type + note
    lines = []
    for e in result["mapping"]:
        line = (
            '    { '
            f'"old_line": {e["old_line"]}, '
            f'"new_lines": {json.dumps(e["new_lines"], ensure_ascii=False)}, '
            f'"type": {json.dumps(e["type"], ensure_ascii=False)}, '
            f'"note": {json.dumps(e["note"], ensure_ascii=False)} '
            '}'
        )
        lines.append(line)
    
    # formating
    body = ",\n".join(lines)
    tail = '\n  ]\n}\n'
    final_text = header + body + tail

    print("done, information mapped to mapping.json")

    # ie 
    # old file
    # FileReader fr = new FileReader(path);
    # FileReader fr = new FileReader(path);
    # public void fileReader(String path){
    # FileReader fr = new FileReader(path);
    # FileReader fr = new FileReader(path);

    # new file
    # FileReader fr = new FileReader(path);
    # FileReader fr = new FileReader(path);
    # public void fileReader(String path){
    # FileReader fr = new FileReader(path);
    # FileReader fr = new FileReader(path);

    # mapping.json
    #{
    # /* header part */
    
    # "mapping": [
    # { "old_line": 1, "new_lines": [1], "type": "unchanged", "note": "" },
    # { "old_line": 2, "new_lines": [2, 3], "type": "unchanged", "note": "split into 2 lines" },
    # { "old_line": 3, "new_lines": [4], "type": "unchanged", "note": "" }

    with open("mapping.json", "w", encoding="utf-8") as f:
        f.write(final_text)
