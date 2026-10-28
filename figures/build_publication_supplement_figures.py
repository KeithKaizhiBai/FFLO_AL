#!/usr/bin/env python3
"""Build method and efficiency diagrams for the public Supplement.

The builder uses only source-code/config metadata and frozen Stage-I counts.
It invokes Figure Engine preflight before rendering and lint after rendering.
No BdG, topology, surrogate training, or active-learning production work is run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures.build_remaining_strict_figures import configure_matplotlib, patch_svg
from figures.figure_engine.builders.figure_build import preflight
from figures.figure_engine.validators.figure_lint import lint_figure


OUT = ROOT / "figures/supplementary_publication_methods"
TEMPLATE = ROOT / "figures/sm_stageiv_numerical_diagnostics/figure_contract.yaml"
SOURCE_CONFIG = ROOT / "downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive/data_archive/stage_i/active_boundary_discovery_rankcap_k3_full_loop_v1/run_config.json"
DATASET = SOURCE_CONFIG.parent / "dataset_iter031.csv"
CANVAS = (1600, 1000)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SPECS = {
    "S1_reliability_flow": {
        "figure_id": "SM-FIG-RELIABILITY-FLOW-V1",
        "message": "The exact oracle separates physical phase assignment from numerical reliability and topology applicability.",
        "labels": ["Exact BdG evaluation", "Reliable label"],
        "source": ROOT / "ml_phase/exact_oracle.py",
    },
    "S2_surrogate_architecture": {
        "figure_id": "SM-FIG-SURROGATE-ARCHITECTURE-V1",
        "message": "Separate five-member MLP ensembles predict observables and thermodynamic phase probabilities for scheduling only.",
        "labels": ["Regression ensemble", "scheduler only"],
        "source": ROOT / "ml_phase/models.py",
    },
    "S3_acquisition_decomposition": {
        "figure_id": "SM-FIG-ACQUISITION-DECOMPOSITION-V1",
        "message": "The production acquisition combines phase, numerical-risk, exploration, and repulsion terms before stochastic microbatch selection.",
        "labels": ["Production acquisition", "microbatch"],
        "source": ROOT / "ml_phase/acquisition.py",
    },
    "S4_sampling_efficiency": {
        "figure_id": "SM-FIG-SAMPLING-EFFICIENCY-V1",
        "message": "The frozen two-dimensional run evaluates 6,880 of 77,361 candidate locations exactly.",
        "labels": ["Candidate universe", "Final exact dataset"],
        "source": DATASET,
    },
}


def _contract(folder: Path, key: str, spec: dict) -> Path:
    base = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    primary = "scientific_content"
    secondary = "interpretation"
    base.update({
        "schema_version": 2,
        "figure_id": spec["figure_id"],
        "figure_type": "supplement",
        "scientific_message": spec["message"],
        "visual_message": spec["message"],
        "primary_object": {"id": primary, "asset_type": "method_diagram", "scientific_role": spec["message"], "visual_role": "primary"},
        "primary_visual_object": {"id": primary, "scientific_role": spec["message"], "expected_rank": 1},
        "secondary_objects": [{"id": secondary, "asset_type": "annotation", "scientific_role": "public interpretation", "visual_role": "secondary"}],
        "tertiary_information": [],
        "secondary_visual_objects": [secondary],
        "tertiary_visual_objects": [],
        "reading_order": [primary, secondary],
        "visual_narrative": {"enabled": True, "narrative_regions": {primary: [primary], secondary: [secondary]}, "scientific_asset_fill_ratio_audits": []},
        "saliency_expectation": {"method": "blurred_region_visual_mass", "expected_ranking": [primary, secondary], "severity": "warning"},
        "forbidden_visual_priorities": ["decorative_machine_learning"],
        "data_sources": [str(spec["source"]), str(SOURCE_CONFIG)],
        "assets": [
            {"id": primary, "type": "method_diagram", "source": str(spec["source"]), "schematic": key != "S4_sampling_efficiency", "bbox": primary},
            {"id": secondary, "type": "annotation", "source": str(spec["source"]), "schematic": True, "bbox": secondary},
        ],
        "layout": {
            "canvas": {"viewbox": [0, 0, *CANVAS], "print_mm": [178.0, 111.25]},
            "template_profile": "supplement_diagnostics",
            "primary_area_fraction_min": 0.45,
            "bounding_boxes": "geometry_manifest.json",
            "alignment_anchors": ["main_flow", "interpretation_footer"],
            "declared_overlays": [], "declared_exceptions": [],
            "area_requirements": [{"id": "primary_area", "object_id": primary, "reference": "canvas", "minimum_fraction": 0.45}],
        },
        "text_budget": {"maximum_words_in_artwork": 180, "maximum_words_per_panel": 180, "minimum_font_size_pt": 7.0, "move_excess_to": "caption"},
        "required_elements": [primary, secondary],
        "required_labels": spec["labels"],
        "forbidden_elements": ["unregistered_color", "text_overlap"],
        "forbidden_claims": ["ML-generated physical label", "wall-time speedup", "autonomous physics solver"],
        "validation_requirements": {
            "hard_fail": True, "require_geometry_manifest": True, "require_editable_svg": True,
            "require_pdf": True, "require_png_preview": True, "require_grayscale_review": True,
            "require_engine_preflight": True, "engine_preflight_record": "engine_preflight.json",
            "strict_palette": True, "prohibited_text": ["PASS", "FAIL", "submission ready"],
        },
        "provenance": {"physics_recomputed": False, "parameters_recorded": True, "commit_hash_when_available": True},
        "approval": {"gate": "GATE_PUBLIC_SUPPLEMENT_METHOD_FIGURES_HUMAN_REVIEW", "state": "HUMAN_WAIT"},
    })
    base.pop("strict_data", None)
    path = folder / "figure_contract.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    geometry = {
        "schema_version": 1,
        "canvas": {"x": 0, "y": 0, "width": 1600, "height": 1000},
        "objects": [
            {"id": primary, "type": "plot", "role": "primary", "parent": "canvas", "visual_weight_object": True, "bbox": {"x": 80, "y": 90, "width": 1440, "height": 720}},
            {"id": secondary, "type": "annotation", "role": "secondary", "parent": "canvas", "visual_weight_object": True, "bbox": {"x": 180, "y": 850, "width": 1240, "height": 80}},
        ],
        "routes": [],
    }
    (folder / "geometry_manifest.json").write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
    return path


def _arrow(ax, a, b, color):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": color, "shrinkA": 5, "shrinkB": 5})


def _box(ax, xy, wh, text, face, edge, fontsize=9, weight="normal"):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor=face, edgecolor=edge, linewidth=1.2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)


def render_s1(ax, c):
    _box(ax, (.04,.57),(.19,.20),"Exact BdG evaluation\n$F(\\Delta,q)$ and spectrum",c["warm"],c["fflo"],10,"bold")
    _box(ax, (.30,.57),(.19,.20),"Adaptive checks\n$q$ window and near-zero $\\Delta$",c["panel"],c["selected"],9)
    _box(ax, (.56,.57),(.17,.20),"Numerical status\nreliable / delicate / rerun",c["cool"],c["uncertainty"],9)
    _box(ax, (.80,.66),(.16,.16),"Reliable label\nNormal / SC / FFLO",c["panel"],c["fflo"],9,"bold")
    _box(ax, (.80,.39),(.16,.16),"Flagged record\nkept separate",c["warm"],c["decision"],9)
    _box(ax, (.56,.20),(.17,.16),"Topology applicable?\nreliable gapped FFLO",c["panel"],c["cfflo"],9)
    _box(ax, (.80,.18),(.16,.16),"$\\nu=0$ or $\\nu=1$\nfrom gap and Pfaffians",c["panel"],c["tfflo"],9)
    for a,b in [((.23,.67),(.30,.67)),((.49,.67),(.56,.67)),((.73,.70),(.80,.74)),((.73,.64),(.80,.47)),((.65,.57),(.65,.36)),((.73,.28),(.80,.26))]: _arrow(ax,a,b,c["axis"])
    ax.text(.5,.91,"Exact-oracle reliability-state flow",ha="center",fontsize=13,fontweight="bold")
    ax.text(.5,.06,"Phase identity and numerical reliability are stored separately; only reliable gapped FFLO states enter topology.",ha="center",fontsize=9)


def render_s2(ax, c):
    _box(ax,(.04,.45),(.15,.24),"Inputs\n$k_BT/t$, $J_A/t$\n(and $\\mu/t$ in 3D)",c["panel"],c["fflo"],9,"bold")
    _box(ax,(.27,.55),(.25,.22),"Regression ensemble\n5 independent MLPs\n2 $\\times$ 64 ReLU",c["cool"],c["cfflo"],9)
    _box(ax,(.27,.25),(.25,.22),"Classifier ensemble\n5 independent MLPs\n2 $\\times$ 64 ReLU",c["cool"],c["cfflo"],9)
    _box(ax,(.61,.55),(.18,.22),"Observable heads\n$\\Delta,q,\\eta,I_c^+,I_c^-$\nmean and spread",c["panel"],c["selected"],9)
    _box(ax,(.61,.25),(.18,.22),"Phase probabilities\nNormal / uniform SC / FFLO\nentropy and margin",c["panel"],c["selected"],9)
    _box(ax,(.86,.39),(.11,.24),"Acquisition\nscheduler only\nnext exact calls",c["warm"],c["decision"],9,"bold")
    for a,b in [((.19,.57),(.27,.66)),((.19,.55),(.27,.36)),((.52,.66),(.61,.66)),((.52,.36),(.61,.36)),((.79,.66),(.86,.55)),((.79,.36),(.86,.47))]: _arrow(ax,a,b,c["axis"])
    ax.text(.5,.91,"Surrogate architecture used by the production scheduler",ha="center",fontsize=13,fontweight="bold")
    ax.text(.5,.07,"Separate models are retrained from accepted exact data; ensemble dispersion supplies uncertainty but never a physical label.",ha="center",fontsize=9)


def render_s3(ax, c):
    labels = [
        ("Phase information","entropy + margin\nregression spread\nNormal--SC and SC--FFLO scores",c["fflo"]),
        ("Numerical risk","$q$-edge risk\nand delicate minima",c["uncertainty"]),
        ("Exploration","extrapolation score\nannealed with iteration",c["selected"]),
    ]
    xs=[.04,.285,.53]
    for x,(title,body,edge) in zip(xs,labels):
        _box(ax,(x,.49),(.20,.28),title+"\n"+body,c["panel"],edge,8.5,"bold" if title=="Phase information" else "normal")
        _arrow(ax,(x+.20,.63),(.77,.63),c["axis"])
    _box(ax,(.77,.51),(.18,.24),"$A_0$ and active pool\ninterior penalty\nrank / quantile filters",c["cool"],c["cfflo"],9,"bold")
    _box(ax,(.35,.16),(.30,.18),"Repelled stochastic microbatch\n$A_{\\rm select}=A_0 R_{\\rm obs}R_{\\rm batch}$",c["warm"],c["decision"],9,"bold")
    _arrow(ax,(.86,.51),(.65,.26),c["axis"])
    ax.text(.5,.91,"Production acquisition and microbatch construction",ha="center",fontsize=13,fontweight="bold")
    ax.text(.5,.06,"Applicability masks and exact-coordinate exclusion act before selection; every selected location is returned to the BdG oracle.",ha="center",fontsize=9)


def render_s4(ax, c):
    candidate=77361; exact=6880; selected=8192; unique_selected=7305
    frac=exact/candidate
    ax.barh([1.0],[candidate],color=c["cool"],edgecolor=c["axis"],height=.24,label="candidate locations")
    ax.barh([.58],[exact],color=c["selected"],edgecolor=c["selected"],height=.24,label="unique exact locations")
    ax.set_xlim(0,candidate*1.05); ax.set_yticks([1.0,.58],["Candidate universe","Final exact dataset"])
    ax.set_xlabel("number of parameter-space locations")
    ax.text(candidate*.985,1.0,f"{candidate:,}",ha="right",va="center",fontsize=10,fontweight="bold")
    ax.text(exact+candidate*.012,.58,f"{exact:,}  ({100*frac:.2f}%)",ha="left",va="center",fontsize=10,fontweight="bold")
    ax.text(.03,.18,f"Selected call records: {selected:,}    Unique selected coordinates: {unique_selected:,}    Exact-location reduction: {candidate/exact:.2f}x",transform=ax.transAxes,fontsize=9)
    ax.text(.03,.08,"The ratio counts locations, not wall time; no matched full-grid timing baseline is available.",transform=ax.transAxes,fontsize=9,color=c["secondary"])
    ax.set_title("Frozen two-dimensional exact-call accounting",fontsize=13,fontweight="bold",pad=16)
    for spine in ("top","right","left"): ax.spines[spine].set_visible(False)
    ax.grid(axis="x",color=c["axis"],alpha=.18,linewidth=.5)


RENDERERS = {"S1_reliability_flow": render_s1, "S2_surrogate_architecture": render_s2, "S3_acquisition_decomposition": render_s3, "S4_sampling_efficiency": render_s4}


def build_one(key: str, spec: dict) -> dict:
    folder = OUT / key; folder.mkdir(parents=True, exist_ok=True)
    contract = _contract(folder,key,spec)
    context = preflight(contract, record_path=folder/"engine_preflight.json")
    c = configure_matplotlib(context)
    fig = plt.figure(figsize=(16,10),dpi=100,facecolor=c["white"])
    ax = fig.add_axes([.06,.10,.88,.82]); ax.set_facecolor(c["white"])
    if key != "S4_sampling_efficiency":
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    RENDERERS[key](ax,c)
    stem=key
    svg=folder/f"{stem}.svg"; pdf=folder/f"{stem}.pdf"; png=folder/f"{stem}.png"
    fig.savefig(svg,format="svg",facecolor=c["white"]); fig.savefig(pdf,format="pdf",facecolor=c["white"]); fig.savefig(png,format="png",facecolor=c["white"],dpi=160)
    plt.close(fig); patch_svg(svg,spec["figure_id"],contract.name)
    with Image.open(png) as image:
        image.convert("L").save(folder/f"{stem}_grayscale.png")
    report=lint_figure(contract_path=contract,svg_path=svg,geometry_path=folder/"geometry_manifest.json",pdf_path=pdf,png_path=png,output_dir=folder)
    if report["status"] != "pass": raise RuntimeError(f"Figure Engine lint failed for {key}")
    return {"key":key,"figure_id":spec["figure_id"],"pdf":str(pdf),"png":str(png),"svg":str(svg),"sha256":sha256(pdf),"lint":"pass"}


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True)
    records=[build_one(key,spec) for key,spec in SPECS.items()]
    (OUT/"figure_manifest.json").write_text(json.dumps({"physics_recomputed":False,"figures":records},indent=2)+"\n",encoding="utf-8")
    pd.DataFrame([{"candidate_grid_size":77361,"unique_exact_coordinates":6880,"selected_call_records":8192,"unique_selected_coordinates":7305,"sampling_fraction":6880/77361,"exact_call_reduction_factor":77361/6880,"boundary_enrichment_factor":"unavailable"}]).to_csv(OUT/"sampling_efficiency.csv",index=False)
    print(json.dumps({"status":"pass","figures":records},indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
