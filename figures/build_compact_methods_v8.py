"""Layout-only S2/S3 candidates: unchanged v7 text, fonts and style tokens."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.build_readable_supplement_methods import (
    plt, Text, Image, yaml, draw, configure_matplotlib, patch_svg,
    preflight, lint_figure,
)

OUT = ROOT / "figures/supplementary_publication_methods_v8"


def main():
    records = []
    for key in ("S2_surrogate_architecture", "S3_acquisition_decomposition"):
        parent = ROOT / "figures/supplementary_publication_methods_v7" / key
        folder = OUT / key
        folder.mkdir(parents=True, exist_ok=True)
        contract = yaml.safe_load((parent / "figure_contract.yaml").read_text())
        contract["figure_id"] = contract["figure_id"].replace("V7", "V8")
        contract["layout"]["canvas"]["print_mm"] = [164, 75]
        contract["provenance"]["layout_parent"] = str(parent.relative_to(ROOT))
        path = folder / "figure_contract.yaml"
        path.write_text(yaml.safe_dump(contract, sort_keys=False))
        geometry = json.loads((parent / "geometry_manifest.json").read_text())
        geometry["objects"] = [x for x in geometry["objects"] if x["type"] != "text"]
        geometry_path = folder / "geometry_manifest.json"
        geometry_path.write_text(json.dumps(geometry, indent=2))
        ctx = preflight(path, record_path=folder / "engine_preflight.json")
        colors = configure_matplotlib(ctx)
        fig = plt.figure(figsize=(164/25.4, 75/25.4), dpi=150, facecolor=colors["white"])
        draw(key, fig.add_axes([.02, .02, .96, .96]), colors)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for i, t in enumerate(fig.findobj(Text)):
            if not t.get_visible() or not t.get_text().strip():
                continue
            b = t.get_window_extent(renderer)
            geometry["objects"].append({
                "id": f"text-{i}", "type": "text", "role": "body", "parent": "canvas",
                "bbox": {"x": b.x0/fig.bbox.width*1600, "y": (fig.bbox.height-b.y1)/fig.bbox.height*1000,
                         "width": b.width/fig.bbox.width*1600, "height": b.height/fig.bbox.height*1000},
                "font_size_pt": t.get_fontsize(), "font_family": "DejaVu Sans",
                "allow_overlap_with": ["scientific_content"],
            })
        geometry_path.write_text(json.dumps(geometry, indent=2))
        for ext in ("svg", "pdf", "png"):
            fig.savefig(folder / f"{key}.{ext}", format=ext, dpi=400)
        plt.close(fig)
        patch_svg(folder / f"{key}.svg", contract["figure_id"], path.name)
        Image.open(folder / f"{key}.png").convert("L").save(folder / f"{key}_grayscale.png")
        result = lint_figure(contract_path=path, svg_path=folder/f"{key}.svg",
                            geometry_path=geometry_path, pdf_path=folder/f"{key}.pdf",
                            png_path=folder/f"{key}.png", output_dir=folder)
        if result["status"] != "pass":
            raise RuntimeError(f"Engine rejected {key}")
        records.append({"key": key, "native_mm": [164, 75], "fonts_unchanged": True,
                        "sha256": hashlib.sha256((folder/f"{key}.pdf").read_bytes()).hexdigest(),
                        "approval": "HUMAN_WAIT"})
    (OUT / "figure_manifest.json").write_text(json.dumps({"figures": records, "physics_recomputed": False}, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
