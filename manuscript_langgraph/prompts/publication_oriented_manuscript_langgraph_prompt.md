You are working inside the exported Active Learning project package.

Your task is to build and initialize a publication-oriented manuscript LangGraph workflow
for this project.

This is NOT a request to immediately write the paper.

The goal is to construct an evidence-first manuscript generation system that can reliably transform:

    frozen scientific artifacts
    ↓
    claims
    ↓
    figures
    ↓
    Results/Methods text
    ↓
    Supplementary information
    ↓
    LaTeX manuscript

with full provenance and consistency control.

==================================================
0. Project context
==================================================

This project developed a reliability-aware active-learning framework
for discovering thermodynamic and topology-resolved FFLO phase diagrams.

The final scientific story should focus on:

Main manuscript:

    1.
    Closed-loop active learning for expensive BdG phase-diagram exploration.

    2.
    Reliable exact oracle design:
        - robust q-window expansion
        - free-energy based phase labeling
        - basin-level refinement
        - numerical reliability control

    3.
    Autonomous reconstruction of:
        - normal
        - uniform superconducting
        - FFLO
        - topology-resolved cFFLO/tFFLO structures

    4.
    Extension toward higher-dimensional parameter spaces if scientifically justified.


Do NOT frame the manuscript primarily as:

    - AI coding workflow
    - ChatGPT/Codex development history
    - engineering diary
    - failed attempts timeline


AI-assisted workflow may appear only as:

    Supplementary methodology / research workflow discussion.

==================================================
1. First perform artifact audit
==================================================

Before building manuscript graph:

Inspect the exported package.

Locate:

    reports
    audit PDFs
    final datasets
    convergence reports
    phase maps
    topology maps
    figure sources
    plotting scripts
    LaTeX files
    previous manuscript drafts
    AI workflow documents
    LangGraph-related files


Generate:

    manuscript_artifact_manifest.csv

Each artifact must contain:

    artifact_id
    path
    type
    source_stage
    run_id
    scientific_role
    provenance_status
    reproducibility_status
    possible_usage:
        main_text
        supplementary
        discussion
        archive_only


Do not write manuscript text before this audit.

==================================================
2. Establish manuscript scope
==================================================

Create:

    manuscript_scope.yaml


The current recommended scope:

Main text:

    Story:
        Reliability-aware active learning reconstructs
        thermodynamic and topology-resolved FFLO phase diagrams.

Include:

    Stage I:
        active-learning discovery of thermodynamic phases

    Stage II:
        exact oracle reliability improvements

    Stage III:
        topology-aware phase discovery and convergence


Optional:

    Stage IV:
        higher-dimensional parameter extension

Only include if:
    evidence is sufficiently strong.


Do NOT make unsupported claims:

Forbidden unless explicitly proven:

    "fully autonomous discovery of arbitrary phases"

    "general AI scientist"

    "complete 3D phase diagram convergence"

    "learned acquisition superiority"


Stage V/V6 ideas:

    learned acquisition
    graph world model
    ground-up discovery

should be treated as:

    future direction
    discussion/outlook

unless separate evidence gate passes.


==================================================
3. Build evidence-first manuscript graph
==================================================

Create a LangGraph workflow.

The graph should NOT start from sections.

It should start from evidence.


Required nodes:


--------------------------------------------------
Node 1:
artifact_provenance_audit
--------------------------------------------------

Input:

    project files


Output:

    verified artifact registry


--------------------------------------------------
Node 2:
claim_freeze
--------------------------------------------------

Create:

    claim_registry.yaml


Every claim requires:

    claim_id

    statement

    scientific_status:
        frozen_supported
        supported_with_caveat
        provisional
        discussion_only
        future_work


    evidence_ids

    allowed_language

    forbidden_language

    limitations


Example:

Claim:

"Active learning reconstructs thermodynamic FFLO phase boundaries."

Allowed:

    reconstructs
    identifies
    converges within studied parameter range


Forbidden:

    discovers all possible phases


--------------------------------------------------
Node 3:
figure_evidence_contract
--------------------------------------------------

Every figure must have:

    figure_id

    scientific purpose

    source artifact

    supported claims

    panels

    caption requirements

    prohibited interpretations


--------------------------------------------------
Node 4:
results_by_figure_writer
--------------------------------------------------

This is the central writing node.

Input:

    figure contract
    evidence
    frozen claims


Output:

    Results paragraph draft


Each paragraph must include:

    observation

    interpretation

    evidence source

    figure reference

    caveats


Separate:

    what is directly observed

    what is scientific interpretation


--------------------------------------------------
Node 5:
methods_writer
--------------------------------------------------

Generate Methods based on final production workflow.

Include:

    BdG oracle

    free-energy minimization

    phase criterion

    q optimization

    topology criterion

    acquisition function

    uncertainty control

    convergence criteria


Do not include development history.

Failures and debugging go to Supplement.


--------------------------------------------------
Node 6:
supplement_builder
--------------------------------------------------

Supplement should contain:


S1:
Active-learning algorithm evolution


S2:
Exact oracle improvements:

    q-window expansion

    local refinement

    K3 basin ranking

    label closure correction


S3:
Acquisition details


S4:
Numerical reliability audits


S5:
AI-assisted research workflow

S6:
Software / LangGraph architecture if useful


--------------------------------------------------
Node 7:
intro_citation_loop
--------------------------------------------------

Adapt QAHE3D-style introduction workflow.

Use only after:

    claims frozen

    Results structure stable


Tasks:

    background claim map

    literature retrieval

    claim-citation matching

    citation placement audit

    Introduction blueprint

    Introduction draft


Do not let Introduction introduce claims unsupported by Results.


--------------------------------------------------
Node 8:
discussion_scope_guard
--------------------------------------------------

Check:

    overclaiming

    AI exaggeration

    unsupported novelty

    future directions mixed with results


Especially monitor:

    learned acquisition

    autonomous discovery

    new phase discovery


--------------------------------------------------
Node 9:
cross_section_consistency_audit
--------------------------------------------------

Check consistency between:

    Abstract

    Introduction

    Results

    Methods

    Discussion

    Supplement


Verify:

    every main claim has evidence

    every figure is referenced

    terminology consistent


--------------------------------------------------
Node 10:
latex_integration
--------------------------------------------------

Prepare:

    manuscript/

including:

    main.tex

    supplementary.tex

    figures/

    bibliography/

    tables/


==================================================
4. Figure workflow and regeneration
==================================================

Important:

This project may require figure regeneration.

Do not assume existing figures are publication-ready.

For every figure:

Audit:

    resolution

    font size

    axis labels

    color consistency

    vector compatibility

    scientific clarity


If necessary:

create:

    figure_redesign_request.yaml


containing:

    current figure problem

    scientific purpose

    required redesign

    source data


Then generate improved figures using:

    existing scripts

or

    new reproducible plotting scripts.


Requirements:

    figures must be reproducible from frozen data.

Never manually edit final figures without source script.


Store:

    figures/source/

    figures/generated/

    figures/final/


==================================================
5. Recommended manuscript figure structure
==================================================


Propose:


Figure 1:

Problem and active-learning framework

Include:

    BdG oracle

    phase criterion

    active-learning loop


Figure 2:

Thermodynamic phase discovery

Include:

    initial sampling

    acquisition evolution

    final phase map

    convergence


Figure 3:

Topology-aware discovery

Include:

    Pfaffian criterion

    cFFLO/tFFLO identification

    topology convergence


Figure 4:

Extension/generalization

Possible:

    3D parameter extension

or:

    efficiency benchmark

Choose based on evidence.


Do not force a figure if evidence is insufficient.


==================================================
6. Supplementary structure
==================================================


Recommended:


Supplementary Note 1:
Physical model and BdG formulation


Supplementary Note 2:
Exact oracle reliability


Supplementary Note 3:
Active-learning acquisition details


Supplementary Note 4:
Convergence and uncertainty audits


Supplementary Note 5:
Additional phase diagrams


Supplementary Note 6:
AI-assisted scientific workflow


Supplementary Note 7:
Software architecture


==================================================
7. QAHE3D reuse policy
==================================================


Reuse only:

    intro_citation_loop concepts

    citation audit logic

    governance concepts


Do NOT copy:

    QAHE3D manuscript assumptions

    section structure blindly

    unsupported PRL templates


Adapt everything to Active Learning evidence.


==================================================
8. Human approval gates
==================================================


Implement gates:


GATE_SCOPE_APPROVAL

before manuscript generation


GATE_CLAIM_APPROVAL

before final wording


GATE_FIGURE_APPROVAL

before LaTeX integration


GATE_FINAL_MANUSCRIPT_APPROVAL


The system should stop at gates requiring scientific judgment.


==================================================
9. Initial execution plan
==================================================


Do NOT write the paper yet.


First execute:

Phase 1:

    artifact audit


Phase 2:

    manuscript scope proposal


Phase 3:

    claim registry draft


Phase 4:

    figure evidence map


Phase 5:

    LangGraph implementation plan


Phase 6:

    choose one vertical slice:

        Stage III topology result

or

        Stage II oracle reliability result


and demonstrate:

    evidence
    figure
    caption
    Results paragraph
    audit


Only after vertical slice approval:

expand to full manuscript.


==================================================
10. Final report
==================================================


Return:

    artifact audit summary

    manuscript scope recommendation

    claim registry status

    figure plan

    LangGraph architecture

    reused QAHE3D components

    newly implemented components

    figure regeneration needs

    vertical slice proposal

    next execution steps


Begin with:

    repository audit
    scientific scope assessment
    manuscript workflow design

Do not generate unsupported manuscript text before evidence contracts exist.
