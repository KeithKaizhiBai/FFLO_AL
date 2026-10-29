"""Portable reader-export guards; no physics calculations or artifact writes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "clean_manuscript_exports",
    ROOT / "manuscript_langgraph/scripts/export_clean_manuscripts.py",
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
VERSIONS = tuple(f"v{number}" for number in range(1, 10))
INCLUDE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
GRAPHIC = re.compile(
    r"\\includegraphics(?:\[[^\]]*\])?\{(?:\\detokenize\{([^{}]+)\}|([^{}]+))\}"
)
BIBLIOGRAPHY = re.compile(r"\\bibliography\{([^{}]+)\}")
XR = re.compile(r"\\externaldocument(?:\[[^\]]*\])?\{([^{}]+)\}")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expanded_source(path):
    """Read this project's static includes using the original TeX working dir."""
    working_directory = path.parent

    def expand(file, stack):
        assert file not in stack, f"Cyclic include: {file}"
        text = file.read_text(encoding="utf-8")

        def include(match):
            target = working_directory / match[1]
            if not target.suffix:
                target = target.with_suffix(".tex")
            return expand(target, (*stack, file))

        return INCLUDE.sub(include, text)

    return expand(path, ())


def tex_dependencies(path):
    """Return exactly the local TeX files reachable from one compilation entry."""
    working_directory = path.parent
    found = set()

    def visit(file):
        assert file.is_file(), f"Missing TeX dependency: {file}"
        assert file.resolve().is_relative_to(working_directory.resolve())
        if file in found:
            return
        found.add(file)
        for name in INCLUDE.findall(file.read_text(encoding="utf-8")):
            target = working_directory / name
            if not target.suffix:
                target = target.with_suffix(".tex")
            visit(target)

    visit(path)
    return found


def original_dependencies(documents):
    dependencies = {}
    for tex, _ in documents.values():
        source = ROOT / tex
        text = expanded_source(source)
        for match in GRAPHIC.finditer(text):
            original = Path(match[1] or match[2])
            if not original.is_absolute():
                original = source.parent / original
            target = "figures/" + original.name
            if target in dependencies:
                assert digest(dependencies[target]) == digest(original)
            dependencies[target] = original
        for group in BIBLIOGRAPHY.findall(text):
            for name in group.split(","):
                original = source.parent / (name.strip().removesuffix(".bib") + ".bib")
                if original.name in dependencies:
                    assert digest(dependencies[original.name]) == digest(original)
                dependencies[original.name] = original
    return dependencies


def test_catalog_keeps_verified_source_pdf_pairings():
    catalog = exporter.catalog()
    assert tuple(catalog) == VERSIONS
    expected = {
        "v1": ("prl_integrated_cited/main.tex", "manuscript_prl_integrated_cited_v1/manuscript_prl_integrated_cited_candidate.pdf"),
        "v2": ("prl_integrated_cited_v2/main.tex", "manuscript_prl_integrated_cited_v2/manuscript_prl_integrated_cited_v2_candidate.pdf"),
        "v3": ("integrated_cited_stageiv_revision_v3/main.tex", "manuscript_integrated_cited_stageiv_revision_v3/manuscript_candidate.pdf"),
        "v4": ("public_reproducible_v4/main.tex", "manuscript_public_reproducible_v4/revised_manuscript_candidate.pdf"),
        "v5": ("scientific_depth_v5/main_A.tex", "manuscript_scientific_depth_v5/manuscript_scientific_depth_candidate.pdf"),
        "v6": ("scientific_depth_single_column_v6/main_barrier.tex", "manuscript_scientific_depth_single_column_v6/manuscript_single_column_candidate.pdf"),
    }
    for version, (tex, pdf) in expected.items():
        assert catalog[version]["manuscript"] == (
            "manuscript_variants/" + tex, "output/pdf/" + pdf,
        )
    assert "supplement" not in catalog["v1"]
    assert "supplement" not in catalog["v2"]
    assert "supplement" not in catalog["v6"]  # No distinct v6 SM was generated.
    for version, directory in {
        "v7": "manuscript_synchronized_review_v7",
        "v8": "manuscript_reader_revision_v8",
        "v9": "manuscript_author_revision_v9",
    }.items():
        for name, job in {
            "manuscript": "manuscript_double_column",
            "review": "manuscript_single_column_review",
            "supplement": "supplement",
        }.items():
            base = f"output/pdf/{directory}/"
            assert catalog[version][name] == (base + f"source/{job}.tex", base + f"{job}.pdf")


@pytest.mark.parametrize("version", VERSIONS)
def test_export_contains_only_required_tex_pdf_figure_and_bib_files(version):
    documents = exporter.catalog()[version]
    directory = ROOT / exporter.DEST / version
    paths = list(directory.rglob("*"))
    assert directory.is_dir() and not directory.is_symlink()
    assert not any(path.is_symlink() for path in paths)
    files = {str(path.relative_to(directory)) for path in paths if path.is_file()}
    assert all(Path(name).suffix in {".tex", ".pdf", ".bib"} for name in files)
    expected = {name + ".pdf" for name in documents}
    for name in documents:
        expected.update(str(path.relative_to(directory)) for path in tex_dependencies(directory / (name + ".tex")))
    expected.update(original_dependencies(documents))
    assert files == expected


@pytest.mark.parametrize("version", VERSIONS)
def test_saved_pdfs_figures_and_bibliography_are_byte_identical(version):
    documents = exporter.catalog()[version]
    directory = ROOT / exporter.DEST / version
    for name, (_, original_pdf) in documents.items():
        assert digest(directory / (name + ".pdf")) == digest(ROOT / original_pdf)
    for relative, original in original_dependencies(documents).items():
        assert digest(directory / relative) == digest(original), relative


@pytest.mark.parametrize("version", VERSIONS)
def test_audit_hashes_cover_all_transitive_tex_and_binary_inputs(version):
    documents = exporter.catalog()[version]
    audit = json.loads((ROOT / exporter.AUDIT / (version + ".json")).read_text())
    expected = set(original_dependencies(documents).values())
    for tex, pdf in documents.values():
        expected.update(tex_dependencies(ROOT / tex))
        expected.add(ROOT / pdf)
    expected_hashes = {
        str(path.relative_to(ROOT)): digest(path) for path in expected
    }
    assert audit["input_hashes"] == expected_hashes
    directory = ROOT / exporter.DEST / version
    assert audit["export_hashes"] == {
        str(path.relative_to(directory)): digest(path)
        for path in directory.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("version", VERSIONS)
def test_all_static_dependencies_resolve_inside_the_export(version):
    directory = ROOT / exporter.DEST / version
    for name in exporter.catalog()[version]:
        entry = directory / (name + ".tex")
        tex_dependencies(entry)
        text = expanded_source(entry)
        assert not INCLUDE.search(text)
        assert r"\graphicspath" not in text
        assert "/Users/" not in text
        for match in GRAPHIC.finditer(text):
            relative = Path(match[1] or match[2])
            assert not relative.is_absolute() and ".." not in relative.parts
            assert (directory / relative).is_file()
        for group in BIBLIOGRAPHY.findall(text):
            for bib in group.split(","):
                assert (directory / (bib.strip() + ".bib")).is_file()
        for external in XR.findall(text):
            assert (directory / (external + ".tex")).is_file()


@pytest.mark.parametrize("version", ("v7", "v8", "v9"))
def test_cross_document_references_follow_exported_job_names(version):
    directory = ROOT / exporter.DEST / version
    for job in ("manuscript", "review"):
        assert XR.findall((directory / (job + ".tex")).read_text()) == ["supplement"]
    assert XR.findall((directory / "supplement.tex").read_text()) == ["manuscript"]


@pytest.mark.parametrize("version", ("v7", "v8", "v9"))
def test_bibtex_detection_reads_nested_bibliography_but_not_sm_notes(version):
    directory = ROOT / exporter.DEST / version
    assert exporter.needs_bibtex(directory / "manuscript.tex", directory)
    assert exporter.needs_bibtex(directory / "review.tex", directory)
    assert not exporter.needs_bibtex(directory / "supplement.tex", directory)


@pytest.fixture
def minimal_export(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.tex").write_text("A fixed manuscript.\n", encoding="utf-8")
    (source / "main.pdf").write_bytes(b"fixed original PDF bytes")
    documents = {"manuscript": ("source/main.tex", "source/main.pdf")}
    return tmp_path, documents


def test_export_preserves_nested_tex_boundaries_and_hashes(minimal_export):
    root, documents = minimal_export
    section = root / "source/sections/statement.tex"
    section.parent.mkdir()
    content = "First paragraph.\n\nSecond paragraph.\n"
    section.write_text(content, encoding="utf-8")
    (root / "source/main.tex").write_text(
        "Before.\n\\input{sections/statement.tex}\nAfter.\n", encoding="utf-8",
    )
    result = exporter.export_version(root, "v1", documents)
    directory = root / exporter.DEST / "v1"
    assert (directory / "manuscript.tex").read_text() == (
        "Before.\n\\input{tex/sections/statement.tex}\nAfter.\n"
    )
    assert (directory / "tex/sections/statement.tex").read_text() == content
    assert result["input_hashes"]["source/sections/statement.tex"] == digest(section)


def test_identical_export_is_idempotent_without_replacing_existing_files(minimal_export):
    root, documents = minimal_export
    first = exporter.export_version(root, "v1", documents)
    directory = root / exporter.DEST / "v1"
    before = {path.name: (path.stat().st_ino, path.stat().st_mtime_ns) for path in directory.iterdir()}
    assert exporter.export_version(root, "v1", documents) == first
    assert before == {path.name: (path.stat().st_ino, path.stat().st_mtime_ns) for path in directory.iterdir()}


@pytest.mark.parametrize("changed", ("destination", "source"))
def test_changed_export_is_rejected_without_overwriting_any_file(minimal_export, changed):
    root, documents = minimal_export
    exporter.export_version(root, "v1", documents)
    directory = root / exporter.DEST / "v1"
    target = directory / "manuscript.tex" if changed == "destination" else root / "source/main.tex"
    target.write_text("User-owned subsequent edit.\n", encoding="utf-8")
    before = exporter.snapshot(directory)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        exporter.export_version(root, "v1", documents)
    assert exporter.snapshot(directory) == before


def test_symlink_destination_is_rejected_without_touching_target(minimal_export):
    root, documents = minimal_export
    destination = root / exporter.DEST / "v1"
    destination.parent.mkdir(parents=True)
    target = root / "user-owned"
    target.mkdir()
    (target / "keep.tex").write_text("Preserve me.\n", encoding="utf-8")
    destination.symlink_to(target, target_is_directory=True)
    before = exporter.snapshot(target)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        exporter.export_version(root, "v1", documents)
    assert destination.is_symlink() and exporter.snapshot(target) == before
