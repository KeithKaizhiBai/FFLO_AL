# Upstream provenance

Adaptation date: 2026-08-07

Upstream project: `/Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/QAHE3D`

The upstream project is not a Git checkout, so a commit identifier is not
available. File-level SHA-256 hashes are recorded instead:

```text
9fbbb500a263e17c320809461e2a3be99e7b70ac160781fe32302ac867ca435a  intro_citation_langgraph/review_agent/subgraphs/intro_citation.py
31030655cd2ec4fc39d90b8319274aace6d06d8e1f9702e22dd167fd5fc025f3  intro_citation_langgraph/review_agent/state.py
8f28a946929766cc5be5dfa98b34131ec542c8a476d60b3895a1e187ac92da76  intro_citation_langgraph/manual.md
32bfc13009c5b8a022a8dea51c7a796ab43dc8d5812fa1772240eec5b6419ce1  agents/orchestration.py
9ecb6f1a0a9255c1eabc556030a5f66f0a14d20c9e6d77b7bf9186e5e1701a46  docs/PRL_manu_prompt.md
df33835ac839dbe6c9c5e4dc3afeb110890c1258d85a9af2af856baa7089e2e4  docs/PRL_manu_script2_prompt.md
```

## Adaptation boundary

The QAHE3D executable source was not copied wholesale because it is specialized
to QAHE3D claims, paths, domain policies, and retrieval artifacts. The reusable
contract was adapted locally as the namespaced `intro_citation_loop` node:
claim map, permission-gated retrieval, version/full-text resolution,
claim-citation mapping, placement audit, paragraph blueprint, and human gates.

The Active Learning adapter adds a stronger prerequisite: approved scope and
frozen Results claims. There is no runtime import from QAHE3D and no two-way
synchronization.

