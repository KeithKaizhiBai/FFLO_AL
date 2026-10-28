# Compact thermodynamic landscape input audit

Date: 2026-08-26

Scope: read-only audit of the current v3 master and landscape input. No master Figure 1 file was modified.

## Machine-measured current geometry

- Declared landscape bbox: `{'x': 385, 'y': 208, 'width': 805, 'height': 300, 'parent': 'central', 'scientific_content_bbox': {'x': 455, 'y': 228, 'width': 665, 'height': 260}}`
- Actual embedded colored-surface bbox: `{'x': 455, 'y': 228, 'width': 665, 'height': 260}`
- Source tight surface PNG: `567 x 273 px`
- Projected aspect ratio: `2.076923`
- Alpha foreground fraction inside source bbox: `0.279706`
- Foreground fraction relative to declared landscape: `0.200253`
- Internal whitespace fraction: `0.799747`
- Three-state projected triangle fraction: `0.026717`
- Projection: `perspective_default`; elev `30°`; azim `-55°`
- 3D box aspect: `[5.0, 1.0, 0.46]`
- Lighting: `{'azimuth_deg': 310, 'altitude_deg': 42, 'hillshade_vertical_exaggeration': 1.12}`

The current foreground mask and marker projection fail the new compact-terrain contract. This is an asset-level result, not a failure of the frozen physics or topology definitions.

## Inputs and hashes

| Input | SHA-256 |
|---|---|
| `figures/fig1_engine_v3/fig1_engine_v3.pdf` | `b1138f2b314443ff874092d4639a0e14c9d370dbd2b9ae7a0bbdf89bd907cda1` |
| `figures/fig1_engine_v3/fig1_engine_v3.svg` | `e7bcef3f2912e56719aa4b3c56b4b6b82f23b0f66430b7ddcbd35045ee90857e` |
| `figures/fig1_engine_v3/build_assets_v3.py` | `459dfe1ede254739c6cddb5d753f83278e0f7113403fb90afa28d44dc4aa9f49` |
| `figures/fig1_engine_v3/asset_versions/thermodynamic_landscape_v3/asset.svg` | `bb999873d85168243243e907a2ab7d33b0fa85e6b72bc19faf81685e5a6eae13` |
| `figures/fig1_engine_v3/asset_versions/thermodynamic_landscape_v3/asset.png` | `8bf9868ab41f1c84201aa1bf3e79a4ef1e4a41cf1a6a9e6d19cc1ceba91f2c02` |
| `figures/fig1_engine_v3/fig1_contract.yaml` | `f572d7af25a683869c0e819ab36990a30778ed4b392376e016810b7b0194d6b9` |
| `figures/figure_engine/constitution/visual_narrative_rules.yaml` | `e450dcfe2acb35f1d0b099aefc9fe10f814c86ef8dceeb06a220dc4e939b99ff` |
| `figures/figure_engine/constitution/asset_rules.yaml` | `396a600e8c4c798a6bfb21f19b394bfe2dd8acbb496d7a6c6d6a7aeff2d1cd38` |
| `figures/figure_engine/constitution/validation_rules.yaml` | `9b9416de08e8beaee0efd04930a54e24b4d0508e548c307b22e603bd52305eff` |
| `figures/fig1_engine_v3/geometry_manifest.json` | `7aeea2205c84892199f22ad2b8909fdcb77c0cd982e4713b5d583ff89b4a2302` |
| `figures/fig1_engine_v3/saliency_audit.json` | `4b7216e806bbcacf8bcb57676440b9aa958fb7fd60df9597465679f036d55604` |
| `figures/fig1_engine_v3/visual_narrative_report.json` | `ef3b77b0f9b99729e0f1fa470520fd8511e0faf5dedff54c00a5afc92665b6c1` |

No exact BdG calculation was run.
