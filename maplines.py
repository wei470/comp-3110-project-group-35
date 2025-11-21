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

    # Step 3: 预处理上下文/SimHash/BOW
    # 假设只上下两行 windows = 2
    old_ctx = [bow_context(old_norm, i, windows) for i in range(len(old_norm))]
    new_ctx = [bow_context(new_norm, i, windows) for i in range(len(new_norm))]
    
    # convert into simhash64 map
    # ie
    # old_ctx[2] = "int b = 2; return sum;"
    # new_ctx[2] = "int b = 3; return sum;"
    # old_shctx[2] = 1010101011...
    # new_shctx[2] = 1010110011...
    # diff will be 3~4 bit

    old_sh, new_sh = [simhash64(s) for s in old_norm], [simhash64(s) for s in new_norm]
    old_shctx, new_shctx = [simhash64(s) for s in old_ctx], [simhash64(s) for s in new_ctx]
    
    # new_ctx[2] = "int b = 3; return sum;"
    # -> new_bowctx = {
    # {
    # 'int': 1,
    # 'b': 1,
    # '3': 1,
    # 'return': 1,
    # 'sum': 1
    # }

    new_bowctx = [Counter(token_iter(s)) for s in new_ctx]

    
    # merge (lo ~ hi) into one line
    # ie
    # 0: "int a = 1;"
    # 1: "int b = 2;"
    # 2: "int sum = a + b;"
    # 3: "return sum;"

    # span_text(2, 2) = "int sum = a + b;"
    # span_text(1, 3) = "int b = 2; int sum = a + b; return sum;"

    def span_text(lo: int, hi: int) -> str:
        return " ".join(new_norm[lo:hi+1])
    
    # add counter from (lo ~ hi)
    # ie
    # new_ctx[0] = "int b = 2;"
    # new_ctx[1] = "int a = 1; int sum = a + b;"
    # new_ctx[2] = "int b = 2; return sum;"
   
    # new_bowctx[0] = Counter({'int':1,'b':1,'=':1,'2':1})
    # new_bowctx[1] = Counter({'int':2,'a':2,'sum':1,'=':1,'b':1})
    # new_bowctx[2] = Counter({'int':1,'b':1,'return':1,'sum':1})

    # span_ctx_counter(0, 0) = Counter({
    #    'int':1, 
    #    'b':1, 
    #    '=':1, 
    #    '2':1
    # })

    # span_ctx_counter(0, 2) = new_bowctx[0] + new_bowctx[1] + new_bowctx[2] = Counter({
    #    'int': 1+2+1 = 4,
    #    'b':   1+1+1 = 3,
    #    'a':   0+2+0 = 2,
    #    'sum': 0+1+1 = 2,
    #    '=':   1+1+0 = 2,
    #    'return': 1
    # })
    
    def span_ctx_counter(lo: int, hi: int) -> Counter:
        c = Counter()
        for t in range(lo, hi+1):
            c.update(new_bowctx[t])
        return c

    mapping = {}
    N = len(new_norm)

    #如果这一行 old[i] 和 new[j] 是 difflib 找到的完全匹配，直接用，跳过 step-3。
    #因为完全相同，不需要任何分析
    for oi, line in enumerate(old_norm):
        if oi in anchors:
            mapping[oi] = anchors[oi]
            continue

    # for each index oi in old
    # search for each index oj in new, evaluate score
    # find top "k" with largest score

    # ie |new| = 5
    # in old[0]
    # compare hashcode of old[0] and new[0...5]
    # select top k smallest.
    
    # ie
    # old_sh     = [12,  25,  59]  
    # old_shctx  = [30,  44,  21]
    # new_sh     = [20, 27, 60, 5]
    # new_shctx  = [40, 45, 20, 3]
    # 只有行内容 → 不够准确
    # 行内容 + 上下文 → 才能找到真正对应的行
    # result = [0, 1, 2, 3]
    # result = [0, 1] (if k =2)

        base = sorted(
            range(N),
            key=lambda nj: hamming(old_sh[oi], new_sh[nj]) + hamming(old_shctx[oi], new_shctx[nj])
        )[:k]

        # store a set of (lo,hi): (from lo to hi)
        cand_spans = set()

        # ie 
        # at old[0], |new| = 10，base= {0,7,9}，maxspan = 2
        # span = 1: cand_spans = {(0,0), (7,7), (9,9)}
        # span = 2: cand_spans = {
        # (0,1), (-1,0)
        # (7,8), (6,7), 
        # (9,10), (8,9)}
        # total = {(-1,0),(0,0),(0,1),(6,7),(7,7),(7,8),(8,9),(8,9),(9,9),(9,10)}

        for j in base:
            for span in range(1, maxspan + 1):
                lo1, hi1 = j, min(N - 1, j + span - 1)
                cand_spans.add((lo1, hi1))
                
                lo2, hi2 = max(0, j - (span - 1)), j
                cand_spans.add((lo2, hi2))
        
        # filer, makke sure 0 ≤ lo ≤ hi < N
        # in example, the total has (-1,0), (9,10), which violate 0 ≤ lo and hi < N remove them from Canadidates
        # cand_spans = {(0,0),(0,1),(6,7),(7,7),(7,8),(8,9),(8,9),(9,9)}

        cand_spans = [(lo, hi) for (lo, hi) in cand_spans if 0 <= lo <= hi < N]
        cand_spans.sort()

        # step 4: For each old[i]，select best cand_spans with highest score 
        # convert [old[i] - windows - old[i] + windows] line into Counter
        bow_src = Counter(token_iter(old_ctx[oi]))
        
        best_score, best_span = -1.0, None
        
        # ie. cand_spans = {(0,0),(0,1),(6,7),(7,7),(7,8),(8,9),(8,9),(9,9)}
        for (lo, hi) in cand_spans:
            new_text = span_text(lo, hi)

            # line = old_norm[oi]
            content_s = edit_sim(line, new_text)

            #原old + windows 对比 cand_spans Counter的cosine value
            ctx_s = cosine_sim(bow_src, span_ctx_counter(lo, hi))

            # ppt #34 
            score = 0.6 * content_s + 0.4 * ctx_s
            
            # record best span for each old[i]
            if score > best_score:
                best_score = score 
                best_span = (lo, hi)

        # for each old[i]，compare every pair in cand_span to find score = 0.6 * content_s + 0.4 * ctx_s
        # find best_span with biggest span_score
        if best_span and best_score >= Tmid:
            lo, hi = best_span
            mapping[oi] = list(range(lo, hi + 1))


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

        