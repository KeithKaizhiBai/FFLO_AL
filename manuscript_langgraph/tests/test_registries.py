from __future__ import annotations

import json
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "registries" / name).read_text(encoding="utf-8"))


def test_required_claims_and_statuses_exist():
    claims = {item["claim_id"]: item for item in load("claim_registry.yaml")["claims"]}
    assert set(claims) == {
        "CLM-2D-THERMO",
        "CLM-ORACLE-RELIABILITY",
        "CLM-2D-TOPOLOGY",
        "CLM-3D-EXTENSION",
        "CLM-LEARNED-ACQUISITION",
        "CLM-GROUNDUP-WORLD-MODEL",
    }
    assert claims["CLM-3D-EXTENSION"]["scientific_status"] == "supported_with_caveat"
    assert claims["CLM-3D-EXTENSION"]["destination"] == "main_text_candidate_figure_4"
    assert claims["CLM-LEARNED-ACQUISITION"]["scientific_status"] == "discussion_only"
    assert claims["CLM-GROUNDUP-WORLD-MODEL"]["destination"] == "outlook"


def test_every_claim_has_registered_evidence():
    claims = load("claim_registry.yaml")["claims"]
    evidence = {item["evidence_id"] for item in load("evidence_registry.yaml")["evidence"]}
    for claim in claims:
        assert set(claim["evidence_ids"]) <= evidence


def test_claim_registry_is_explicitly_approved_with_modifications():
    registry = load("claim_registry.yaml")
    assert registry["approval_status"] == "approved_with_modifications"
    assert registry["scope_id"] == "AL_FFLO_PRL_RELIABILITY_AWARE_3D_V2"
    assert all(claim["human_approval_state"] == "approved" for claim in registry["claims"])


def test_main_figures_have_caveats_and_prohibitions():
    figures = load("figure_registry.yaml")["figures"]
    assert len(figures) == 4
    for figure in figures:
        assert figure["required_caveats"]
        assert figure["prohibited_claims"]


def test_artifact_manifest_contract_and_usage_vocabulary():
    manifest = ROOT / "audits" / "manuscript_artifact_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "artifact_id",
        "path",
        "type",
        "source_stage",
        "run_id",
        "scientific_role",
        "provenance_status",
        "reproducibility_status",
        "possible_usage",
    }
    assert rows
    assert required <= set(rows[0])
    allowed = {"main_text", "supplementary", "discussion", "archive_only"}
    for row in rows:
        assert set(row["possible_usage"].split("|")) <= allowed
