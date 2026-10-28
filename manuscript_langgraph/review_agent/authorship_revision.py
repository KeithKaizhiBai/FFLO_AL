"""Author identity and administrative availability child of the v8 graph."""
import difflib
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from pypdf import PdfReader
from review_agent.synchronized_review import sha, write, record, JOBS

PARENT = "output/pdf/manuscript_reader_revision_v8"
OUTPUT = "output/pdf/manuscript_author_revision_v9"
RUN = "active_learning_author_revision_v9"
AUTHOR = "Kai-Zhi Bai"
EMAIL = "kzbai@connect.hku.hk"
AFFILIATION = "Department of Physics, The University of Hong Kong, Pokfulam Road, Hong Kong, China"
AUTHOR_TEX = "\\author{" + AUTHOR + "}\n\\email{" + EMAIL + "}\n\\affiliation{" + AFFILIATION + "}\n"
REPOSITORY = "https://github.com/KeithKaizhiBai/FFLO_AL"
AVAILABILITY_EDITS = {
    "discussion.tex": (
        "No public repository or persistent identifier has been established for this review version.",
        r"The associated project repository is \url{" + REPOSITORY + "}. Its README records the available code, data and manuscript artifacts and their limitations; no archival DOI is claimed."),
    "supplement.tex": (
        "Neither the archived scientific package nor these manuscript sources have a confirmed public repository URL or persistent identifier; public access and long-term deposition remain to be arranged by the authors.",
        r"The associated project repository is \url{" + REPOSITORY + "}; its README identifies available artifacts and reproduction limitations. Repository commits do not retroactively establish the missing historical executable bindings, and no archival DOI is claimed."),
}


def update_availability(text, filename):
    if filename in AVAILABILITY_EDITS:
        old, new = AVAILABILITY_EDITS[filename]
        if text.count(old) != 1:
            raise RuntimeError("Unexpected availability anchor: " + filename)
        return text.replace(old, new)
    return text


def repository_discovery(state):
    root = Path(state["project_root"])
    parent = root / PARENT
    for job in JOBS:
        if not (parent / (job + ".pdf")).is_file():
            raise RuntimeError("Missing v8 baseline: " + job)
    hashes = {str(p.relative_to(root)): sha(p) for p in parent.rglob("*") if p.is_file()}
    manifest = json.loads((parent / "input_manifest.json").read_text())
    manifest.update(reviewed_parent=PARENT, parent_hashes=hashes,
                    author_identity={"name": AUTHOR, "email": EMAIL, "affiliation": AFFILIATION,
                                     "authorized_date": "2026-10-28"})
    record(root / OUTPUT / "input_manifest.json", manifest)
    return {"repository_discovery_complete": True, "parent_source_hashes_before": hashes,
            "variant_output_root": str(root / OUTPUT), "run_id": RUN}


def build_sources(state):
    root = Path(state["project_root"])
    parent, out = root / PARENT, root / OUTPUT
    source = out / "source"
    shutil.copytree(parent / "source", source, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("built", "built_logs", "__pycache__"))
    write(source / "authors.tex", AUTHOR_TEX)
    write(source / "authors_sm.tex", AUTHOR_TEX.replace("\\email{" + EMAIL + "}\n", ""))
    for filename in AVAILABILITY_EDITS:
        path = source / "sections" / filename
        write(path, update_availability(path.read_text(), filename))
    metadata = (source / "metadata.tex").read_text()
    metadata = metadata.replace("% Author/affiliation block intentionally absent: not yet confirmed.",
                                "% Author identity explicitly supplied on 2026-10-28; see authors.tex.\n"
                                "\\hypersetup{pdfauthor={Kai-Zhi Bai}}")
    write(source / "metadata.tex", metadata)
    for filename, anchor in (("main_content.tex", r"\title{\ManuscriptTitle}"),
                             ("supplement.tex", r"\title{Supplemental Material: \ManuscriptTitle}")):
        text = (source / filename).read_text()
        if text.count(anchor) != 1:
            raise RuntimeError("Unexpected title anchor in " + filename)
        text = text.replace(anchor, anchor + "\n\\input{authors.tex}\n")
        if filename == "supplement.tex":
            text = text.replace(r"\input{authors.tex}", r"\input{authors_sm.tex}")
            text = text.replace("tightenlines,floatfix]", "tightenlines,superscriptaddress,floatfix]")
            # This standalone SM has no bibliography, where REVTeX normally
            # emits the email note. Keep its contact address visible explicitly.
            text = text.replace(r"\maketitle", r"\maketitle" + "\n"
                                + r"\begin{center}\small\href{mailto:" + EMAIL + "}{" + EMAIL
                                + r"}\end{center}" + "\n")
        write(source / filename, text)
    for directory in ("tables", "figures"):
        shutil.copytree(parent / directory, out / directory, dirs_exist_ok=True)
    author_header = f"Author: {AUTHOR}\n\nEmail: {EMAIL}\n\nAffiliation: {AFFILIATION}\n\n"
    for job in JOBS:
        text = (parent / (job + ".md")).read_text()
        filename = "supplement.tex" if job == "supplement" else "discussion.tex"
        text = update_availability(text, filename)
        first, rest = text.split("\n", 1)
        write(out / (job + ".md"), first + "\n\n" + author_header + rest)
    review = (parent / "AUTHOR_REVIEW.md").read_text()
    write(out / "AUTHOR_REVIEW.md", "# v9 identity update — 2026-10-28\n\n"
          "Name, email and affiliation are confirmed and inserted. This resolves only the identity portion of R13. "
          "Funding, scientific choices and figure approval are not inferred. The following is the preserved v8 review record.\n\n" + review)
    write(out / "decision_log.md", "# Decision log — 2026-10-28\n\n"
          "- Add the user-supplied author, email and affiliation to all three documents.\n"
          "- Preserve v8, scientific content, bibliography and figure binaries; update only two administrative availability sentences to the user-specified repository.\n"
          "- Retain all unresolved science/figure gates; authorship is not scientific approval.\n"
          "- Target upload repository: https://github.com/KeithKaizhiBai/FFLO_AL. Upload status is recorded separately.\n"
          "- No physics recomputation or new literature retrieval.\n")
    write(out / "README.md", "# Author-identified manuscript candidate v9\n\n"
          f"{AUTHOR} · {EMAIL}\n\n{AFFILIATION}\n\n"
          "Author identity and repository-availability revision of v8, authorized 2026-10-28. Scientific content and artwork are unchanged. "
          "This remains an initial draft, not a scientifically approved or submission-ready version.\n\n"
          "The three PDFs share a source fingerprint. Markdown, tables, figures and a portable LaTeX ZIP accompany them. "
          "Run `python3 source/build.py` with pypdf, pdfLaTeX, BibTeX and REVTeX installed. "
          "The repository workflow is `manuscript_langgraph/scripts/run_single_column_placement.py --profile authorship_v9`.\n\n"
          "Inherited open items: R01/R02/R03 Ansatz/R11/R12, remaining R13 information, R04/R07 historical binding, "
          "Fig.3/4/S5 readability, and main-text float interruptions. See AUTHOR_REVIEW.md.\n")
    differences = []
    for filename in ("authors.tex", "authors_sm.tex", "metadata.tex", "main_content.tex", "supplement.tex", "sections/discussion.tex", "sections/supplement.tex"):
        old = parent / "source" / filename
        differences.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],
                            (source / filename).read_text().splitlines(True),
                            fromfile="v8/"+filename, tofile="v9/"+filename))
    write(out / "source_changes.diff", "".join(differences))
    return {"publication_sources_ready": True, "workflow_status": "authorship_sources_ready"}


def compile_and_select(state):
    out = Path(state["project_root"]) / OUTPUT
    subprocess.run([sys.executable, "build.py"], cwd=out / "source", check=True)
    for job in JOBS:
        shutil.copy2(out / "source/built" / (job+".pdf"), out / (job+".pdf"))
    checks = json.loads((out / "source/built/validation.json").read_text())
    with (out / "tables/figure_placements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["document", "figure", "page", "width_mm", "height_mm"])
        writer.writeheader()
        for job, data in checks.items():
            for i, form in enumerate(data["embedded_forms"], 1):
                writer.writerow({"document": job, "figure": i, "page": form["page"],
                                 "width_mm": form["rendered_width_mm"], "height_mm": form["rendered_height_mm"]})
    return {"revision_compiled": True, "workflow_status": "authorship_compiled"}


def audit_and_gate(state):
    root = Path(state["project_root"])
    parent, out = root / PARENT, root / OUTPUT
    for path, expected in state["parent_source_hashes_before"].items():
        if sha(root/path) != expected:
            raise RuntimeError("v8 input changed: " + path)
    for folder in ("sections", "figures"):
        for p in (parent / "source" / folder).rglob("*"):
            if not p.is_file():
                continue
            target = out / "source" / p.relative_to(parent / "source")
            if folder == "sections" and p.name in AVAILABILITY_EDITS:
                if update_availability(p.read_text(), p.name) != target.read_text():
                    raise RuntimeError("Change beyond administrative availability: " + p.name)
            elif sha(p) != sha(target):
                raise RuntimeError("Scientific content changed: " + str(p))
    if sha(parent / "source/references.bib") != sha(out / "source/references.bib"):
        raise RuntimeError("Bibliography changed")
    for job in JOBS:
        pdf = PdfReader(out / (job+".pdf"))
        text = " ".join(" ".join(p.extract_text() for p in pdf.pages).split())
        for value in (AUTHOR, EMAIL, AFFILIATION):
            if value not in text:
                raise RuntimeError(f"Author identity missing from {job}: {value}")
    with zipfile.ZipFile(out / "latex_source_package.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in (out / "source").rglob("*"):
            if p.is_file() and not {"built", "built_logs", "__pycache__"}.intersection(p.relative_to(out / "source").parts):
                z.write(p, p.relative_to(out))
    result = {"workflow_status": "HUMAN_WAIT", "current_gate": "GATE_READER_REVISION_SCIENCE_CHOICES",
              "gate_status": "HUMAN_WAIT", "author_identity_approved": True, "physics_changed": False,
              "v8_unchanged": True, "scientific_content_figures_bibliography_unchanged": True,
              "administrative_availability_updated": True,
              "single_column_manuscript_pdf": str(out / "manuscript_single_column_review.pdf")}
    record(out / "build_state.json", result)
    record(root / "manuscript_langgraph/runs" / RUN / "final_state.json", result)
    return result
