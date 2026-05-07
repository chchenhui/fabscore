# Supplementary Materials — TRAIL–DR5 In Silico Optimization 

**Paper:** Initiation of Programmed Cell Death in Cancer Stem Cells: In Silico Mutagenesis for Optimized TRAIL–DR5 Binding

This archive is designed so even beginners can reproduce the computational project step by step.
It explains *why* each step is done and uses simple terms.

---

## What this project is about
Scientists are trying to make a special protein (TRAIL) stick better to another protein (DR5).
When they stick together, cancer stem cells can self-destruct.
We are using computers to design versions of TRAIL that stick better.

---

## What's in this folder
- `README.md` — these instructions
- `requirements.txt` — a list of Python tools you need
- `run_analysis.py` — a script that makes graphs from the docking results
- `docking_results.csv` — an example file with docking results (like a scoreboard)
- `pymol_commands.txt` — step-by-step instructions for PyMOL (a protein viewer)
- `apbs_template.in` — a template file for APBS (to see charges on proteins)
- `hdock_submission_instructions.txt` — easy guide to use HDOCK (a website that docks proteins)
- `supplementary_notes.md` — tips, FAQs, and extra explanations
- `docker/Dockerfile` — optional setup file for advanced users
- `LICENSE` — the license
- `CITATION.txt` — how to cite this project

---

## Step-by-Step Instructions (like a recipe!)

### Step 1: Install the tools
1. Download **PyMOL** (protein viewer, like 3D glasses for proteins).
2. Download **APBS** (shows charged areas on proteins).
3. Install Python 3.8 or newer.
4. In a terminal, type:
   ```
   pip install -r requirements.txt
   ```

### Step 2: Get the proteins
1. Go to the Protein Data Bank (www.rcsb.org).
2. Search for **1D4V** → download this (it’s TRAIL).
3. Search for **4N90** → download this (it’s DR5).
4. Put both files in a new folder called `inputs/`.

### Step 3: Prepare proteins in PyMOL
- Run:
  ```
  pymol -cq pymol_commands.txt
  ```
- This will clean up the proteins (remove water, etc.) and save them into an `outputs/` folder.

### Step 4: Mutate TRAIL
- Open PyMOL normally (not in command line).
- Load TRAIL.
- Use the "Mutagenesis Wizard" to change specific amino acids at the binding site.
- Save each mutant as `outputs/TRAIL_mutant_XXX.pdb`.

### Step 5: Dock with HDOCK
1. Go to https://hdock.phys.hust.edu.cn/
2. Upload DR5 as "Receptor".
3. Upload your TRAIL mutant as "Ligand".
4. Click "Submit".
5. After it's done, download the results.

### Step 6: Record the scores
- Find the Docking Score and Confidence in the HDOCK results.
- Add them to `docking_results.csv`.

### Step 7: Make graphs
Run:
```
python run_analysis.py
```
This will create graphs in a folder called `results/`.

---

## How to understand the results
- **Docking Score:** Like golf, lower = better (proteins fit together better).
- **Confidence Score:** Higher is better (the program is more sure about the result).

---

## FAQ
**Q: What if my PyMOL looks different?**  
A: Different versions look slightly different. Look for menus like "Wizard" and "Mutagenesis".

**Q: What if HDOCK won’t accept my files?**  
A: Make sure you used the cleaned PDB files (not the originals with water molecules).

**Q: How do I know if my mutant worked?**  
A: If the Docking Score is lower and Confidence is higher than Wild-Type, your mutant is better.

