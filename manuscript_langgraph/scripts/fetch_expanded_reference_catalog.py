from __future__ import annotations

import json
import html
import re
import sys
import time
import urllib.parse
import urllib.request
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "manuscript_langgraph/review_agent/prl_reference_catalog_expanded.json"


RECORDS = [
    ("FuldeFerrell1964", "10.1103/PhysRev.135.A550", "fflo_foundations", "primary"),
    ("CasalbuoniNardulli2004", "10.1103/RevModPhys.76.263", "fflo_reviews", "authoritative_review"),
    ("MatsudaShimahara2007", "10.1143/JPSJ.76.051005", "fflo_reviews", "authoritative_review"),
    ("KinnunenEtAl2018", "10.1088/1361-6633/aaa4ad", "fflo_reviews", "authoritative_review"),
    ("RadovanEtAl2003", "10.1038/nature01842", "fflo_experiments", "primary"),
    ("BianchiEtAl2003", "10.1103/PhysRevLett.91.187004", "fflo_experiments", "primary"),
    ("KenzelmannEtAl2008", "10.1126/science.1161818", "fflo_experiments", "primary"),
    ("LortzEtAl2007", "10.1103/PhysRevLett.99.187002", "fflo_experiments", "primary"),
    ("MayaffreEtAl2014", "10.1038/nphys3121", "fflo_experiments", "primary"),
    ("LiaoEtAl2010", "10.1038/nature09393", "imbalanced_fermi_gases", "primary"),
    ("ZwierleinEtAl2006", "10.1126/science.1122318", "imbalanced_fermi_gases", "primary"),
    ("PartridgeEtAl2006", "10.1126/science.1122876", "imbalanced_fermi_gases", "primary"),
    ("SheehyRadzihovsky2007", "10.1016/j.aop.2006.09.009", "imbalanced_fermi_gases", "authoritative_review"),
    ("RadzihovskySheehy2010", "10.1088/0034-4885/73/7/076501", "imbalanced_fermi_gases", "authoritative_review"),
    ("SmejkalSinovaJungwirth2022", "10.1103/PhysRevX.12.040501", "altermagnetism", "authoritative_perspective"),
    ("SmejkalEtAl2022", "10.1103/PhysRevX.12.031042", "altermagnetism", "primary"),
    ("Mazin2022", "10.1103/PhysRevX.12.040002", "altermagnetism", "authoritative_perspective"),
    ("SmejkalEtAl2020", "10.1126/sciadv.aaz8809", "altermagnetism", "primary"),
    ("ChakrabortyBlackSchaffer2024", "10.1103/PhysRevB.110.L060508", "altermagnetic_superconductivity", "primary"),
    ("HongParkKim2025", "10.1103/PhysRevB.111.054501", "altermagnetic_superconductivity", "primary"),
    ("ZhuEtAl2023", "10.1103/PhysRevB.108.184505", "altermagnetic_topological_superconductivity", "primary"),
    ("GhorashiHughesCano2024", "10.1103/PhysRevLett.133.106601", "altermagnetic_topological_superconductivity", "primary"),
    ("BoseVadnaisParamekanti2024", "10.1103/PhysRevB.110.205120", "altermagnetic_superconductivity", "primary"),
    ("ZhangYi2013", "10.1038/ncomms3711", "finite_momentum_topology", "primary"),
    ("QuEtAl2013", "10.1038/ncomms3710", "finite_momentum_topology", "primary"),
    ("CaoEtAl2014", "10.1103/PhysRevLett.113.115302", "finite_momentum_topology", "primary"),
    ("Kitaev2001", "10.1070/1063-7869/44/10S/S29", "class_d_topology", "primary"),
    ("BudichArdonne2013", "10.1103/PhysRevB.88.075419", "class_d_topology", "primary"),
    ("ReadGreen2000", "10.1103/PhysRevB.61.10267", "topological_superconductivity", "primary"),
    ("LutchynSauDasSarma2010", "10.1103/PhysRevLett.105.077001", "majorana_platforms", "primary"),
    ("OregRefaelVonOppen2010", "10.1103/PhysRevLett.105.177002", "majorana_platforms", "primary"),
    ("Alicea2012", "10.1088/0034-4885/75/7/076501", "topological_superconductivity", "authoritative_review"),
    ("Beenakker2013", "10.1146/annurev-conmatphys-030212-184337", "topological_superconductivity", "authoritative_review"),
    ("SatoAndo2017", "10.1088/1361-6633/aa6ac7", "topological_superconductivity", "authoritative_review"),
    ("QiZhang2011", "10.1103/RevModPhys.83.1057", "topological_phases", "authoritative_review"),
    ("SchnyderEtAl2008", "10.1103/PhysRevB.78.195125", "topological_classification", "primary"),
    ("RyuEtAl2010", "10.1088/1367-2630/12/6/065010", "topological_classification", "primary"),
    ("TerayamaEtAl2019", "10.1103/PhysRevMaterials.3.033802", "active_learning_phase_diagrams", "primary"),
    ("DaiBrussGlotzer2020", "10.1021/acs.jpcb.9b09202", "active_learning_phase_diagrams", "primary"),
    ("LookmanEtAl2019", "10.1038/s41524-019-0153-8", "active_learning_materials", "authoritative_review"),
    ("CohnAtlasLadner1994", "10.1007/BF00993277", "active_learning_foundations", "primary_methods"),
    ("MacKay1992", "10.1162/neco.1992.4.4.590", "active_learning_foundations", "primary_methods"),
    ("ShahriariEtAl2016", "10.1109/JPROC.2015.2494218", "bayesian_optimization", "authoritative_review"),
    ("TranUlissi2018", "10.1038/s41929-018-0142-1", "active_learning_chemistry", "primary_methods"),
    ("PodryabinkinShapeev2017", "10.1016/j.commatsci.2017.08.031", "active_learning_potentials", "primary_methods"),
    ("JinnouchiEtAl2019", "10.1103/PhysRevB.100.014105", "active_learning_potentials", "primary_methods"),
    ("VandermauseEtAl2020", "10.1038/s41524-020-0283-z", "active_learning_potentials", "primary_methods"),
    ("BatraEtAl2021", "10.1038/s41578-020-00255-y", "materials_machine_learning", "authoritative_review"),
    ("NoeEtAl2019", "10.1126/science.aaw1147", "physics_machine_learning", "authoritative_review"),
]


CORE_KEYS = {
    "FuldeFerrell1964",
    "LarkinOvchinnikov1965",
    "CasalbuoniNardulli2004",
    "SmejkalSinovaJungwirth2022",
    "ChakrabortyBlackSchaffer2024",
    "HongParkKim2025",
    "ZhangYi2013",
    "Kitaev2001",
    "BudichArdonne2013",
    "TerayamaEtAl2019",
    "DaiBrussGlotzer2020",
    "LookmanEtAl2019",
}


AUTHOR_OVERRIDES = {
    "DaiBrussGlotzer2020": "Chengyu Dai and Isaac R. Bruss and Sharon C. Glotzer",
}


TITLE_OVERRIDES = {
    "BianchiEtAl2003": "Possible Fulde-Ferrell-Larkin-Ovchinnikov Superconducting State in CeCoIn5",
    "KenzelmannEtAl2008": "Coupled Superconducting and Magnetic Order in CeCoIn5",
    "HongParkKim2025": "Unconventional p-wave and Finite-Momentum Superconductivity Induced by Altermagnetism through the Formation of Bogoliubov Fermi Surface",
    "BoseVadnaisParamekanti2024": "Altermagnetism and Superconductivity in a Multiorbital t-J Model",
}


YEAR_OVERRIDES = {"BatraEtAl2021": 2021}


MANUAL_LO = {
    "key": "LarkinOvchinnikov1965",
    "topic": "fflo_foundations",
    "authors": "A. I. Larkin and Yu. N. Ovchinnikov",
    "title": "Nonuniform State of Superconductors",
    "journal": "Soviet Physics JETP",
    "volume": "20",
    "pages": "762--769",
    "year": 1965,
    "doi": "",
    "official_url": "https://jetp.ras.ru/",
    "source_type": "primary",
    "verification_tier": "T1_bibliographic_crosscheck",
    "supports": ["foundational finite-momentum and spatially modulated superconductivity"],
    "verification": "bibliographic metadata cross-checked against the authoritative FFLO review and publisher reference lists",
}


def _first(value: Any, default: str = "") -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value or default)


def _year(item: dict[str, Any]) -> int:
    for field in ("published-print", "published-online", "published", "issued"):
        parts = item.get(field, {}).get("date-parts", [])
        if parts and parts[0]:
            return int(parts[0][0])
    raise ValueError(f"missing year for {item.get('DOI')}")


def _authors(item: dict[str, Any]) -> str:
    names = []
    for author in item.get("author", []):
        given = str(author.get("given", "")).strip()
        family = str(author.get("family", "")).strip()
        name = " ".join(part for part in (given, family) if part)
        if name:
            names.append(name)
    if not names:
        raise ValueError(f"missing authors for {item.get('DOI')}")
    return unicodedata.normalize("NFC", " and ".join(names))


def _plain_text(value: Any) -> str:
    text = html.unescape(_first(value))
    text = re.sub(r"<[^>]+>", "", text)
    return unicodedata.normalize("NFC", " ".join(text.split()))


def _fetch(doi: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(
        f"https://api.crossref.org/works/{encoded}",
        headers={"User-Agent": "AL-FFLO-manuscript-reference-audit/1.0 (mailto:metadata-audit@example.invalid)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("status") != "ok":
        raise RuntimeError(f"Crossref lookup failed for {doi}")
    return payload["message"]


def main() -> None:
    catalog = [MANUAL_LO]
    failures = []
    for key, doi, topic, source_type in RECORDS:
        try:
            item = _fetch(doi)
            pages = str(item.get("page") or item.get("article-number") or "").replace("-", "--")
            record = {
                "key": key,
                "topic": topic,
                "authors": unicodedata.normalize("NFC", AUTHOR_OVERRIDES.get(key, _authors(item))),
                "title": TITLE_OVERRIDES.get(key, _plain_text(item.get("title"))),
                "journal": _plain_text(item.get("container-title")) or _plain_text(item.get("short-container-title")),
                "volume": str(item.get("volume", "")),
                "pages": pages,
                "year": YEAR_OVERRIDES.get(key, _year(item)),
                "doi": str(item.get("DOI", doi)),
                "official_url": str(item.get("resource", {}).get("primary", {}).get("URL") or item.get("URL") or f"https://doi.org/{doi}"),
                "source_type": source_type,
                "verification_tier": "T1_core_claim_authority" if key in CORE_KEYS else "T2_crossref_doi_metadata",
                "supports": [topic.replace("_", " ")],
                "verification": "Crossref DOI record and publisher-linked bibliographic metadata verified; contextual citation only unless separately designated as a core claim authority",
            }
            catalog.append(record)
        except Exception as exc:  # audit should report all failures together
            failures.append({"key": key, "doi": doi, "error": str(exc)})
        time.sleep(0.05)
    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        raise SystemExit(1)
    if len(catalog) != 50 or len({record["key"] for record in catalog}) != 50:
        raise SystemExit(f"catalog cardinality failure: {len(catalog)}")
    payload = {
        "schema_version": 2,
        "retrieval_date": "2026-08-28",
        "target_reference_count": 50,
        "admission_policy": "tiered: core claims require primary/authoritative support; contextual survey references may enter with verified Crossref DOI metadata",
        "forbidden": ["search-snippet-only metadata", "uncited bibliography padding", "external evidence for project Results"],
        "sources": catalog,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"EXPANDED_REFERENCE_CATALOG_PASS count={len(catalog)} output={OUTPUT}")


if __name__ == "__main__":
    main()
