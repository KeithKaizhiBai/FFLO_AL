#!/usr/bin/env python3
"""Mechanical reader export of existing manuscript versions; no physics or prose edits.

Version folders contain only required TeX, original compiled PDFs, referenced
figures and bibliography files. All build debris and audit metadata stay outside
the reader folders. Existing files are never silently overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("reports/manuscript")
AUDIT = Path("docs/report_qa/20261029_manuscript_exports")
GRAPHIC = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)(?:\\detokenize\{([^{}]+)\}|([^{}]+))\}")
BIB = re.compile(r"\\bibliography\{([^{}]+)\}")
INPUT = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
EXTERNAL = {"manuscript_double_column": "manuscript", "manuscript_single_column_review": "review"}


def catalog():
    variants, outputs = "manuscript_variants/", "output/pdf/"
    rows = {}
    old = [
        (1, "prl_integrated_cited", "manuscript_prl_integrated_cited_v1", "main.tex", "manuscript_prl_integrated_cited_candidate.pdf", None),
        (2, "prl_integrated_cited_v2", "manuscript_prl_integrated_cited_v2", "main.tex", "manuscript_prl_integrated_cited_v2_candidate.pdf", None),
        (3, "integrated_cited_stageiv_revision_v3", "manuscript_integrated_cited_stageiv_revision_v3", "main.tex", "manuscript_candidate.pdf", "supplement_candidate.pdf"),
        (4, "public_reproducible_v4", "manuscript_public_reproducible_v4", "main.tex", "revised_manuscript_candidate.pdf", "revised_supplement_candidate.pdf"),
        (5, "scientific_depth_v5", "manuscript_scientific_depth_v5", "main_A.tex", "manuscript_scientific_depth_candidate.pdf", "supplement_scientific_depth_candidate.pdf"),
        (6, "scientific_depth_single_column_v6", "manuscript_scientific_depth_single_column_v6", "main_barrier.tex", "manuscript_single_column_candidate.pdf", None),
    ]
    for n, variant, output, tex, pdf, sm in old:
        rows[f"v{n}"] = {"manuscript": (variants + variant + "/" + tex, outputs + output + "/" + pdf)}
        if sm:
            rows[f"v{n}"]["supplement"] = (variants + variant + "/supplement.tex", outputs + output + "/" + sm)
    for n, output in [(7, "manuscript_synchronized_review_v7"), (8, "manuscript_reader_revision_v8"), (9, "manuscript_author_revision_v9")]:
        base = outputs + output + "/"
        rows[f"v{n}"] = {name: (base + "source/" + job + ".tex", base + job + ".pdf") for name, job in
                         [("manuscript", "manuscript_double_column"), ("review", "manuscript_single_column_review"), ("supplement", "supplement")]}
    return rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def copy_dependency(source, destination, root, evidence):
    source = source.resolve()
    if not source.is_relative_to(root.resolve()) or not source.is_file():
        raise ValueError(f"Missing or out-of-project dependency: {source}")
    if destination.exists() and sha(destination) != sha(source):
        raise ValueError(f"Conflicting dependency basename: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    evidence[str(source.relative_to(root.resolve()))] = sha(source)
    if sha(destination) != sha(source):
        raise ValueError(f"Dependency changed during copy: {source}")


def tex_inputs(source, root, base=None, stack=()):
    """Record the static input closure using TeX's entry-directory path rule."""
    source = source.resolve()
    base = base or source.parent
    if source in stack or not source.is_relative_to(root.resolve()):
        raise ValueError(f"Cyclic or out-of-project TeX input: {source}")
    evidence = {str(source.relative_to(root.resolve())): sha(source)}
    text = re.sub(r"(?<!\\)%[^\n]*", "", source.read_text())
    for name in INPUT.findall(text):
        path = base / name
        if not path.suffix:
            path = path.with_suffix(".tex")
        evidence.update(tex_inputs(path, root, base, stack + (source,)))
    return evidence


def needs_bibtex(entry, root):
    return any(BIB.search(re.sub(r"(?<!\\)%[^\n]*", "", (root / path).read_text()))
               for path in tex_inputs(entry, root))


def stage_tex(source, destination, root, evidence, base=None, stack=()):
    """Preserve input boundaries: flattening can change TeX whitespace/layout."""
    source = source.resolve()
    base = base or source.parent
    if source in stack or not source.is_relative_to(base):
        raise ValueError(f"Cyclic or out-of-source TeX input: {source}")
    text = source.read_text()

    def include(match):
        path = base / match[1]
        if not path.suffix:
            path = path.with_suffix(".tex")
        relative = path.resolve().relative_to(base)
        content = stage_tex(path, destination, root, evidence, base, stack + (source,))
        target = destination / "tex" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text() != content:
            raise ValueError(f"Conflicting TeX dependency: {relative}")
        target.write_text(content)
        return match[0].replace(match[1], "tex/" + relative.as_posix(), 1)

    text = INPUT.sub(include, text)

    def graphic(match):
        path = Path(match[2] or match[3])
        path = path if path.is_absolute() else base / path
        copy_dependency(path, destination / "figures" / path.name, root, evidence)
        return match[1] + "figures/" + path.name + "}"

    text = GRAPHIC.sub(graphic, text)
    for group in BIB.findall(text):
        for name in group.split(","):
            path = base / (name.strip().removesuffix(".bib") + ".bib")
            copy_dependency(path, destination / path.name, root, evidence)
        # The current supported sources have a local references.bib only.
        if group != "references":
            raise ValueError("Review nonstandard bibliography path before exporting")
    for old, new in EXTERNAL.items():
        text = re.sub(r"(\\externaldocument(?:\[[^\]]*\])?\{)" + old + r"\}",
                      lambda m: m[1] + new + "}", text)
    if str(root) in text or "/Users/" in text or "\\graphicspath" in text:
        raise ValueError("Unresolved absolute or indirect asset path: " + str(source))
    return text


def export_version(root, version, documents):
    destination = root / DEST / version
    evidence = {}
    with tempfile.TemporaryDirectory(prefix="fflo_reader_export_") as temporary:
        work = Path(temporary)
        for name, (tex, pdf) in documents.items():
            source = root / tex
            evidence.update(tex_inputs(source, root))
            text = stage_tex(source, work, root, evidence)
            (work / (name + ".tex")).write_text(text, encoding="utf-8")
            copy_dependency(root / pdf, work / (name + ".pdf"), root, evidence)
        files = snapshot(work)
        if any(sha(root / path) != digest for path, digest in evidence.items()):
            raise ValueError("An original input changed during export")
        if destination.exists():
            if destination.is_symlink() or any(p.is_symlink() for p in destination.rglob("*")) or snapshot(destination) != files:
                raise ValueError(f"Refusing to overwrite changed reader version: {destination}")
        else:
            shutil.copytree(work, destination)
    return {"documents": documents, "input_hashes": evidence, "export_hashes": files,
            "original_pdfs_preserved_byte_for_byte": True}


def verify_version(root, version, documents):
    """Clean-room rebuild; retain original PDFs, not newly timestamped builds."""
    from pypdf import PdfReader
    from PIL import Image
    source = root / DEST / version
    with tempfile.TemporaryDirectory(prefix="fflo_reader_compile_") as temporary:
        work = Path(temporary)
        for p in source.rglob("*"):
            if p.is_file() and (p.suffix in {".tex", ".bib"} or "figures" in p.relative_to(source).parts):
                target = work / p.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
        jobs = sorted(documents, key=lambda name: (name != "supplement", name))

        def run(command):
            result = subprocess.run(command, cwd=work, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(" ".join(command) + "\n" + result.stdout[-5000:])

        def latex(job):
            # Match the historical TeX Live restricted mode, never unrestricted
            # shell escape. Disabling it changes one v9 SM line's spacing here.
            run(["pdflatex", "-shell-restricted", "-recorder", "-interaction=nonstopmode", "-halt-on-error", job + ".tex"])

        for job in jobs:
            latex(job)
            # REVTeX emits jobNotes bibdata even for an SM with no bibliography.
            # Only explicit bibliography commands in the reached source closure
            # should trigger BibTeX, matching the original paired-build recipe.
            if needs_bibtex(work / (job + ".tex"), work):
                run(["bibtex", job])
        for _ in range(3):
            for job in jobs:
                latex(job)
        checks = {}
        for job in jobs:
            old = root / documents[job][1]
            rebuilt = work / (job + ".pdf")
            pdf, original = PdfReader(rebuilt), PdfReader(old)
            diagnostic = root / "tmp/pdfs/reader_export_checks" / version
            diagnostic.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rebuilt, diagnostic / (job + ".pdf"))
            text = lambda reader: [" ".join((page.extract_text() or "").split()) for page in reader.pages]
            log = (work / (job + ".log")).read_text(errors="replace")
            if re.search(r"undefined references|undefined citations|Citation .+ undefined|Reference .+ undefined", log, re.I):
                raise RuntimeError(f"Unresolved citation or reference: {version}/{job}")
            if text(pdf) != text(original):
                raise RuntimeError(f"Rebuilt pages/text differ from saved PDF: {version}/{job}")
            # Pixel comparison verifies that source packaging and relative paths did not
            # alter figure placement or layout; PDF metadata timestamps may differ.
            for prefix, path in [("old", old), ("new", rebuilt)]:
                run(["pdftoppm", "-r", "72", "-png", str(path), str(work / prefix)])
            old_pages = sorted(work.glob("old-*.png"))
            new_pages = sorted(work.glob("new-*.png"))
            def same_pixels(a, b):
                with Image.open(a) as first, Image.open(b) as second:
                    return first.size == second.size and first.convert("RGB").tobytes() == second.convert("RGB").tobytes()
            pixels_match = len(old_pages) == len(new_pages) and all(same_pixels(a, b) for a, b in zip(old_pages, new_pages))
            # Prefixes are reused for the next document, which may be shorter.
            for p in old_pages + new_pages:
                p.unlink()
            if not pixels_match:
                raise RuntimeError(f"Rendered pages differ: {version}/{job}")
            checks[job] = {"pages": len(pdf.pages), "page_text_identical": True,
                           "all_pages_pixel_identical_72dpi": True, "undefined_references": False}
        return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", choices=list(catalog()), default=list(catalog()))
    parser.add_argument("--verify", action="store_true", help="Recompile exported sources in clean temporary directories")
    args = parser.parse_args()
    (ROOT / AUDIT).mkdir(parents=True, exist_ok=True)
    for version in args.versions:
        documents = catalog()[version]
        result = export_version(ROOT, version, documents)
        if args.verify:
            result["clean_compile"] = verify_version(ROOT, version, documents)
        audit = ROOT / AUDIT / (version + ".json")
        if audit.exists() and not args.verify:
            previous = json.loads(audit.read_text())
            if previous.get("export_hashes") == result["export_hashes"] and "clean_compile" in previous:
                result["clean_compile"] = previous["clean_compile"]
        audit.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"version": version, "files": len(result["export_hashes"]),
                          "verified": "clean_compile" in result}), flush=True)


if __name__ == "__main__":
    main()
