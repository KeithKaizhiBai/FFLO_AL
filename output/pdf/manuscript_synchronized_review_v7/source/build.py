"""Portable paired build. Requires Python 3, pypdf, pdfLaTeX, BibTeX."""
from pathlib import Path
import hashlib
import json
import re
import math
import shutil
import subprocess
import tempfile
from pypdf import PdfReader

JOBS = ("manuscript_double_column", "manuscript_single_column_review", "supplement")


def validate_log(log, single):
    failures = re.findall(r"undefined references|undefined citations|Citation .+ undefined|Reference .+ undefined|multiply defined|Overfull \\hbox|Overfull \\vbox", log, re.I)
    if failures:
        raise RuntimeError("Unresolved references or layout overflow: " + repr(failures))
    text = float(re.search(r"LAYOUT-TEXTWIDTH=([0-9.]+)pt", log)[1])
    col = float(re.search(r"LAYOUT-COLUMNWIDTH=([0-9.]+)pt", log)[1])
    if (abs(text-col) < 0.1) != single:
        raise RuntimeError(f"Column configuration does not match output name: single={single}, text={text}, column={col}")
    return {"textwidth_pt": text, "columnwidth_pt": col, "actual_single_column": single}


def validate_pdf(path, fingerprint):
    pdf = PdfReader(path)
    if fingerprint not in str(pdf.metadata.subject):
        raise RuntimeError("Stale or independently built PDF substituted")
    text = "\n".join(p.extract_text() for p in pdf.pages)
    if "Author list pending" in text or "??" in text:
        raise RuntimeError("Placeholder or unresolved PDF reference")
    return {"pages": len(pdf.pages), "source_fingerprint": fingerprint, "text": text}


def validate_aux(aux, supplement):
    rows = re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux)
    local = [(k,v) for k,v in rows if not k.startswith(("main-", "sm-"))]
    for prefix in ("sec:", "eq:", "fig:", "tab:", "alg:"):
        values = [v for k,v in local if k.startswith(prefix)]
        if len(values) != len(set(values)):
            raise RuntimeError("Repeated automatic numbering: " + prefix)
        if supplement and any(not v.startswith("S") for v in values):
            raise RuntimeError("Missing S numbering: " + prefix)


def validate_figure_size(width_bp, native_width_bp):
    if width_bp/native_width_bp < 0.85:
        raise RuntimeError("Complex artwork reduced below native-width review allowance")


def build():
    src = Path(__file__).resolve().parent
    files = [p for p in src.rglob("*") if p.is_file() and not any(x in p.relative_to(src).parts for x in ("built", "built_logs", "__pycache__"))]
    hashes = {str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="fflo_paired_v7_") as name:
        work = Path(name)
        for p in files:
            q = work / p.relative_to(src); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p,q)
        stamp = '\\hypersetup{pdfsubject={Paired-source-sha256 ' + fingerprint + '}}\n'
        with (work / "preamble.tex").open("a") as f:
            f.write(stamp)

        def latex(job):
            result = subprocess.run(["pdflatex", "-recorder", "-interaction=nonstopmode", "-halt-on-error", job+".tex"], cwd=work, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(job + " failed:\n" + result.stdout[-6000:])

        # Bootstrap the external labels from this same clean source set only.
        latex("supplement")
        for job in JOBS[:2]:
            latex(job)
            bib = subprocess.run(["bibtex",job], cwd=work, capture_output=True, text=True)
            if bib.returncode:
                raise RuntimeError(bib.stdout)
        for _ in range(3):
            for job in ("supplement", *JOBS[:2]):
                latex(job)
        checks = {}
        debug = src / "built_logs"; debug.mkdir(exist_ok=True)
        for job in JOBS:
            shutil.copy2(work/(job+".log"), debug/(job+".log"))
        for job in JOBS:
            log = (work / (job+".log")).read_text(errors="replace")
            checks[job] = validate_log(log, job != JOBS[0])
            validate_aux((work/(job+".aux")).read_text(), job=="supplement")
            pdf = validate_pdf(work / (job+".pdf"), fingerprint)
            checks[job].update({k:v for k,v in pdf.items() if k != "text"})
            (work / (job+".md")).write_text(pdf["text"])
            fls = (work / (job+".fls")).read_text()
            if job != "supplement":
                for n in range(1,5):
                    if f"figures/Fig{n}_candidate.pdf" not in fls:
                        raise RuntimeError(f"Figure {n} not present in actual recorder inputs")
            # Read the actual PDF form resources, not just the LaTeX source.
            forms = []
            for page_i, page in enumerate(PdfReader(work / (job+".pdf")).pages,1):
                objects = page.get("/Resources",{}).get("/XObject",{})
                placements = {}
                def operand(op, args, cm, tm):
                    if op == b'Do' and str(args[0]) in objects:
                        placements[str(args[0])] = list(cm)
                page.extract_text(visitor_operand_before=operand)
                for key, ref in objects.items():
                    obj = ref.get_object()
                    if obj.get("/Subtype") == "/Form":
                        box = [float(v) for v in obj.get("/BBox",[])]; cm = placements[key]
                        width = (box[2]-box[0])*math.hypot(cm[0],cm[1])
                        if job != "supplement": validate_figure_size(width, box[2]-box[0])
                        forms.append({"page": page_i, "name": key, "bbox": box, "matrix": cm,
                                      "rendered_width_mm": width*25.4/72, "rendered_height_mm": (box[3]-box[1])*math.hypot(cm[2],cm[3])*25.4/72})
            checks[job]["embedded_forms"] = forms
        # Publish nothing until every member of the pair/set passed.
        dest = src / "built"; dest.mkdir(exist_ok=True)
        for job in JOBS:
            for ext in ("pdf","aux","bbl","fls","log","md"):
                p=work/(job+"."+ext)
                if p.exists(): shutil.copy2(p,dest/p.name)
        (dest/"validation.json").write_text(json.dumps(checks,indent=2))
        (dest/"input_hashes.json").write_text(json.dumps(hashes,indent=2))
        print(json.dumps(checks,indent=2))


if __name__ == "__main__":
    build()
