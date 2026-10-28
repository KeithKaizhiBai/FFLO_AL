import importlib.util
from pathlib import Path
import pytest
from pypdf import PdfWriter
from review_agent.synchronized_review import verify_figures, replace_once

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("paired_build", ROOT/"manuscript_variants/synchronized_review_v7/build.py")
paired=importlib.util.module_from_spec(spec);spec.loader.exec_module(paired)


def test_approved_figure_control():
    verify_figures(ROOT)


def test_wrong_fig1_fails(tmp_path):
    p=tmp_path/"figures/fig1_engine_v4/fig1_engine_v4.pdf";p.parent.mkdir(parents=True)
    p.write_bytes((ROOT/"figures/fig1_engine_v3/fig1_engine_v3.pdf").read_bytes())
    with pytest.raises(RuntimeError,match="approved artwork"): verify_figures(tmp_path)


def test_stale_supplement_and_control(tmp_path):
    p=tmp_path/"supplement.pdf"; w=PdfWriter();w.add_blank_page(612,792);w.add_metadata({"/Subject":"Paired-source-sha256 current"});w.write(p)
    paired.validate_pdf(p,"current")
    with pytest.raises(RuntimeError,match="Stale"): paired.validate_pdf(p,"new-build")


def test_column_and_reference_negative_controls():
    two="LAYOUT-TEXTWIDTH=510.0pt\nLAYOUT-COLUMNWIDTH=246.0pt"
    paired.validate_log(two,False)
    with pytest.raises(RuntimeError,match="Column"):paired.validate_log(two,True)
    with pytest.raises(RuntimeError,match="Unresolved"):paired.validate_log(two+"\nundefined references",False)


def test_duplicate_and_disabled_section_numbering():
    paired.validate_aux(r"\newlabel{sec:a}{{S1}{1}}\newlabel{sec:b}{{S2}{2}}",True)
    with pytest.raises(RuntimeError,match="Repeated"):paired.validate_aux(r"\newlabel{sec:a}{{S3}{1}}\newlabel{sec:b}{{S3}{2}}",True)
    with pytest.raises(RuntimeError,match="Missing S"):paired.validate_aux(r"\newlabel{sec:a}{{}{1}}",True)


def test_complex_figure_size_negative_control():
    paired.validate_figure_size(508,504.57)
    with pytest.raises(RuntimeError,match="Complex artwork"):paired.validate_figure_size(245,504.57)


def test_scope_limited_replacement():
    assert replace_once("one phrase", "phrase", "word")=="one word"
    with pytest.raises(RuntimeError):replace_once("phrase phrase","phrase","word")
