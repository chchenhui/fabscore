#!/usr/bin/env python3
"""
Quick LaTeX compiler for debugging.

Usage:
  python tools/compile_latex.py --dir results/minimal/20250908_205835/latex
  # or
  python tools/compile_latex.py --tex results/minimal/20250908_205835/latex/template.tex

It runs: pdflatex (1 pass) -> bibtex (if needed) -> pdflatex (2 passes)
Prints exit codes and extracts key errors/warnings from .log and .blg.
"""
import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys
import re


def run(cmd, cwd):
    res = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return res.returncode, res.stdout, res.stderr


def detect_main_tex(latex_dir: str) -> str:
    cand = osp.join(latex_dir, "template.tex")
    if osp.exists(cand):
        return cand
    for f in os.listdir(latex_dir):
        if f.endswith(".tex"):
            return osp.join(latex_dir, f)
    raise FileNotFoundError("No .tex file found in directory: " + latex_dir)


def need_bibtex(tex_path: str) -> bool:
    try:
        with open(tex_path, "r", encoding="utf-8") as f:
            src = f.read()
        if "\\bibliography" in src or "\\addbibresource" in src:
            return True
    except Exception:
        pass
    # Also check aux if present
    aux = osp.splitext(tex_path)[0] + ".aux"
    if osp.exists(osp.join(osp.dirname(tex_path), osp.basename(aux))):
        try:
            with open(aux, "r", encoding="utf-8", errors="replace") as f:
                s = f.read()
            if "\\bibdata" in s or "\\bibstyle" in s:
                return True
        except Exception:
            pass
    return False


def summarize_logs(log_path: str, blg_path: str | None = None):
    def tail(path, n=120):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return ""

    log = tail(log_path, 400)
    blg = tail(blg_path, 200) if blg_path and osp.exists(blg_path) else ""

    # Extract key issues
    errors = re.findall(r"^! .*", log, flags=re.M)
    undef = re.findall(r"Undefined control sequence.*", log)
    missing_math = re.findall(r"Missing \$ inserted.*", log)
    unicode_err = re.findall(r"Unicode character .* not set up.*", log)
    overfull = len(re.findall(r"Overfull \\hbox", log))
    underfull = len(re.findall(r"Underfull \\hbox", log))

    print("\n=== LaTeX log tail (last ~400 lines) ===")
    print(log)
    if blg:
        print("\n=== BibTeX log tail (last ~200 lines) ===")
        print(blg)

    print("\n=== Summary ===")
    print(f"Errors: {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(" -", e)
        if len(errors) > 10:
            print(" - …")
    print(f"Undefined control seq lines: {len(undef)}")
    print(f"Missing $ inserted lines: {len(missing_math)}")
    print(f"Unicode errors: {len(unicode_err)}")
    print(f"Overfull hbox: {overfull}, Underfull hbox: {underfull}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir", help="Path to LaTeX working directory containing .tex", default=None
    )
    ap.add_argument(
        "--tex", help="Path to main .tex file (overrides --dir)", default=None
    )
    ap.add_argument(
        "--engine",
        help="LaTeX engine to use",
        choices=["pdflatex", "xelatex", "lualatex"],
        default="pdflatex",
    )
    args = ap.parse_args()

    if args.tex:
        tex_path = osp.abspath(args.tex)
        latex_dir = osp.dirname(tex_path)
    elif args.dir:
        latex_dir = osp.abspath(args.dir)
        tex_path = detect_main_tex(latex_dir)
    else:
        print("Provide --dir or --tex")
        sys.exit(2)

    if shutil.which(args.engine) is None:
        print(
            f"{args.engine} not found. Install TeX Live / MacTeX and ensure it is on PATH."
        )
        sys.exit(1)

    main_stem = osp.splitext(osp.basename(tex_path))[0]
    print(f"Working dir: {latex_dir}")
    print(f"Main .tex:   {osp.basename(tex_path)}")

    # First pass
    code, out, err = run(
        [args.engine, "-interaction=nonstopmode", osp.basename(tex_path)], latex_dir
    )
    print(f"{args.engine} pass 1 -> exit {code}")

    # BibTeX if needed
    if need_bibtex(tex_path) and shutil.which("bibtex") is not None:
        code_bib, out_bib, err_bib = run(["bibtex", main_stem], latex_dir)
        print(f"bibtex -> exit {code_bib}")
    else:
        code_bib = None

    # Two more passes
    code2, out2, err2 = run(
        [args.engine, "-interaction=nonstopmode", osp.basename(tex_path)], latex_dir
    )
    print(f"{args.engine} pass 2 -> exit {code2}")
    code3, out3, err3 = run(
        [args.engine, "-interaction=nonstopmode", osp.basename(tex_path)], latex_dir
    )
    print(f"{args.engine} pass 3 -> exit {code3}")

    # Summarize logs
    log_path = osp.join(latex_dir, main_stem + ".log")
    blg_path = osp.join(latex_dir, main_stem + ".blg")
    summarize_logs(log_path, blg_path)


if __name__ == "__main__":
    main()
