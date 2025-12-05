# Source Code Line Tracker (LHDiff Implementation)

**COMP 3110 Project - Group 35**

This repository contains a Python-based tool for tracking source code lines across different versions of a file. It implements a hybrid approach (inspired by LHDiff) combining **SimHash**, **Levenshtein Distance**, and **Contextual Cosine Similarity** to accurately map modified, deleted, and added lines between pair of two files.

## 📂 Project Structure

The codebase is organized into the following modules:

* **`maplines.py`**: The **core engine**. It contains the implementation of the line mapping algorithm, including:
    * Line normalization and tokenization.
    * SimHash calculation (64-bit) for fast candidate filtering.
    * Levenshtein distance for content similarity.
    * Cosine similarity for context analysis (using Bag-of-Words).
* **`mappingpy_script.py`**: A **batch processing script**. It automatically iterates through the dataset folder (`evaluate_set`), processes pairs of files (`old_x.txt` vs `new_x.txt`), and generates JSON mapping results
ie: for old_1.txt and new_1.txt, it will generate a result mapping_1.json in folder(`evaluate_set`).

* **`true_ground.py`**: Generates **Ground Truth** data. It reads the source files and creates a baseline mapping (based on exact matching) to serve as a reference for evaluation.
ie: for old_1.txt and new_1.txt, it will generate a result pair1.json in folder(`evaluate_set`).

* **`compare_map.py`**: The **evaluation script**. It compares the tool's output (`mapping_x.json`) against the ground truth (`pairx.txt`) and calculates metrics like Correctness, Spurious matches, and Changes and output to terminal, at the same time it will generate a text file: (`compare.txt`) in main folder.

---

## 🚀 Algorithm Overview

Our approach follows a multi-step pipeline to ensure accuracy and performance:

1.  **Preprocessing**: Lines are normalized (whitespace removal) and tokenized.
2.  **Exact Matching**: Identifies identical lines immediately to use as "anchors."
3.  **SimHash Filtering**: Uses 64-bit SimHash to quickly find candidate lines in the new file that are roughly similar to the old line, reducing the search space.
4.  **Hybrid Scoring**: Calculates a combined score for candidate spans based on:
    * *Content Similarity* (0.6 weight): Using Edit Distance (Levenshtein).
    * *Context Similarity* (0.4 weight): Using Cosine Similarity of the surrounding lines (window size = 4).
5.  **Span Optimization**: Detects line splits and merges by analyzing variable-length spans (controlled by `MAXSPAN = 2`).

---

## 🛠️ Setup & Usage

### Prerequisites
* Python 3.x
* Standard libraries only (`os`, `json`, `argparse`, `difflib`, `math`, `hashlib`, `collections`). No `pip install` required.

### 1. Running the Tool on a Single Pair
To map lines between two specific files:
1. place old.txt, new.txt file into main folder
run: (`python maplines.py old.txt new.txt`)

### 2. Running the Tool on sets of pair file
1. place old_x.txt, new_x.txt into folder evaluate_set
2. run (`python mappingpy_script.py`) it will generate mapping_x.json file in evaluate_set folder
3. run (`python true_ground.py`) it will generate pairx.txt file in evaluate_set folder
4. run (`python compare_map.py`) it will compare the difference between mapping_x.json and pairx.txt to produce compare.txt file in main folder

### All comment in one line to generate correct rate for sets of pair files: 
python mappingpy_script.py 
python true_ground.py 
python compare_map.py
