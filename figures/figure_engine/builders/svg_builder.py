#!/usr/bin/env python3
"""Small editable-SVG primitives that consume canonical engine rules."""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from .figure_build import BuildContext
from .rule_loader import RuleBundle, load_rule_bundle


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


class SVGBuilder:
    def __init__(self, width: int, height: int, *, figure_id: str, contract_path: str,
                 rules: RuleBundle | None = None, minimum_font_size_pt: float = 7.0,
                 allowed_extra_hex: list[str] | None = None):
        self.rules = rules or load_rule_bundle()
        self.minimum_font_size_pt = float(minimum_font_size_pt)
        self.allowed_colors = self._approved_colors()
        self.allowed_colors.update(value.upper() for value in (allowed_extra_hex or []))
        self.geometry: list[dict] = []
        self.root = ET.Element(_tag("svg"), {
            "width": str(width), "height": str(height), "viewBox": f"0 0 {width} {height}",
            "data-figure-id": figure_id, "data-contract": contract_path,
            "data-engine-schema": "1",
        })
        ET.SubElement(self.root, _tag("rect"), {
            "x": "0", "y": "0", "width": str(width), "height": str(height),
            "fill": self.rules["palette"]["neutral"]["white"],
            "data-role": "canvas-background",
        })

    @classmethod
    def from_context(cls, width: int, height: int, context: BuildContext) -> "SVGBuilder":
        return cls(
            width, height, figure_id=context.contract.figure_id,
            contract_path=str(context.contract.path), rules=context.rules,
            minimum_font_size_pt=context.contract["text_budget"]["minimum_font_size_pt"],
            allowed_extra_hex=context.contract["color_semantics"].get("allowed_extra_hex", []),
        )

    def _approved_colors(self) -> set[str]:
        colors = {str(value).upper() for value in self.rules["palette"]["neutral"].values()}
        for family in self.rules["palette"]["families"].values():
            colors.update(str(value).upper() for value in family.values())
        for scale in self.rules["palette"].get("continuous", {}).values():
            colors.update(str(value).upper() for value in scale)
        return colors

    @staticmethod
    def _localname(node: ET.Element) -> str:
        return node.tag.rsplit("}", 1)[-1]

    @staticmethod
    def _prefix_ids(root: ET.Element, prefix: str) -> None:
        mapping: dict[str, str] = {}
        for node in root.iter():
            current = node.get("id")
            if current:
                mapping[current] = f"{prefix}-{current}"
                node.set("id", mapping[current])
        for node in root.iter():
            for key, value in list(node.attrib.items()):
                for old, new in mapping.items():
                    value = value.replace(f"url(#{old})", f"url(#{new})").replace(f"#{old}", f"#{new}")
                node.set(key, value)
            if SVGBuilder._localname(node) == "style" and node.text:
                for old, new in mapping.items():
                    node.text = node.text.replace(f"url(#{old})", f"url(#{new})")

    def group(self, group_id: str, *, role: str) -> ET.Element:
        return ET.SubElement(self.root, _tag("g"), {"id": group_id, "data-role": role})

    def title(self, value: str, description: str) -> None:
        ET.SubElement(self.root, _tag("title")).text = value
        ET.SubElement(self.root, _tag("desc")).text = description

    def rect(self, parent: ET.Element, *, x: float, y: float, width: float, height: float,
             fill: str, stroke: str = "none", stroke_width: float = 0,
             radius: float = 0, element_id: str | None = None, opacity: float = 1.0) -> ET.Element:
        for color in (fill, stroke):
            if color != "none" and color.upper() not in self.allowed_colors:
                raise ValueError(f"unapproved color token: {color}")
        attributes = {
            "x": str(x), "y": str(y), "width": str(width), "height": str(height),
            "fill": fill.upper() if fill != "none" else fill,
            "stroke": stroke.upper() if stroke != "none" else stroke,
            "stroke-width": str(stroke_width), "opacity": str(opacity),
        }
        if radius:
            attributes.update({"rx": str(radius), "ry": str(radius)})
        if element_id:
            attributes["id"] = element_id
        return ET.SubElement(parent, _tag("rect"), attributes)

    def path(self, parent: ET.Element, d: str, *, stroke: str = "none", fill: str = "none",
             stroke_width: float = 1.0, opacity: float = 1.0, element_id: str | None = None) -> ET.Element:
        for color in (fill, stroke):
            if color != "none" and not color.startswith("url(") and color.upper() not in self.allowed_colors:
                raise ValueError(f"unapproved color token: {color}")
        attributes = {"d": d, "stroke": stroke, "fill": fill, "stroke-width": str(stroke_width),
                      "opacity": str(opacity), "stroke-linecap": "round", "stroke-linejoin": "round"}
        if element_id:
            attributes["id"] = element_id
        return ET.SubElement(parent, _tag("path"), attributes)

    def nested_svg(self, parent: ET.Element, source: str | Path, *, asset_id: str,
                   x: float, y: float, width: float, height: float) -> ET.Element:
        parsed = ET.parse(source).getroot()
        self._prefix_ids(parsed, asset_id)
        nested = ET.SubElement(parent, _tag("svg"), {
            "id": f"asset-{asset_id}", "x": str(x), "y": str(y), "width": str(width),
            "height": str(height), "viewBox": parsed.get("viewBox", f"0 0 {width} {height}"),
            "preserveAspectRatio": "xMidYMid meet", "overflow": "visible", "data-role": "scientific-asset",
        })
        for child in parsed:
            if self._localname(child) != "metadata":
                nested.append(copy.deepcopy(child))
        return nested

    def text(self, parent: ET.Element, x: float, y: float, value: str, *, role: str, size_pt: float,
             bbox: dict[str, float], fill: str | None = None, anchor: str = "start",
             text_id: str | None = None, weight: int | str = 400) -> ET.Element:
        if float(size_pt) < self.minimum_font_size_pt:
            raise ValueError(f"font size {size_pt:g} pt is below contract minimum {self.minimum_font_size_pt:g} pt")
        if not {"x", "y", "width", "height"}.issubset(bbox) or float(bbox["width"]) <= 0 or float(bbox["height"]) <= 0:
            raise ValueError("every text object requires a measured positive bbox")
        selected_fill = (fill or self.rules["palette"]["neutral"]["body"]).upper()
        if selected_fill not in self.allowed_colors:
            raise ValueError(f"unapproved color token: {selected_fill}")
        node = ET.SubElement(parent, _tag("text"), {
            "x": str(x), "y": str(y), "data-role": role, "font-size": f"{size_pt}pt",
            "font-family": self.rules["fonts"]["sans"]["preferred"],
            "font-stretch": "normal", "text-anchor": anchor,
            "fill": selected_fill,
            "data-bbox": ",".join(str(bbox[key]) for key in ("x", "y", "width", "height")),
            "font-weight": str(weight),
        })
        if text_id:
            node.set("id", text_id)
        node.text = value
        self.geometry.append({
            "id": node.get("id") or f"text-{len(self.geometry) + 1}", "type": "text", "role": role,
            "parent": parent.get("id", "root"), "bbox": dict(bbox),
            "font_family": self.rules["fonts"]["sans"]["preferred"],
            "font_stretch": "normal", "font_size_pt": float(size_pt),
            "transform": {"scale_x": 1, "scale_y": 1},
        })
        return node

    @staticmethod
    def _cubic(points: list[list[float]], s: float) -> tuple[float, float]:
        p0, p1, p2, p3 = points
        u = 1 - s
        return (
            u ** 3 * p0[0] + 3 * u * u * s * p1[0] + 3 * u * s * s * p2[0] + s ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * s * p1[1] + 3 * u * s * s * p2[1] + s ** 3 * p3[1],
        )

    @staticmethod
    def _tangent(points: list[list[float]], s: float) -> tuple[float, float]:
        p0, p1, p2, p3 = points
        u = 1 - s
        return (
            3 * u * u * (p1[0] - p0[0]) + 6 * u * s * (p2[0] - p1[0]) + 3 * s * s * (p3[0] - p2[0]),
            3 * u * u * (p1[1] - p0[1]) + 6 * u * s * (p2[1] - p1[1]) + 3 * s * s * (p3[1] - p2[1]),
        )

    @staticmethod
    def _smoothstep(value: float) -> float:
        return 3 * value * value - 2 * value * value * value

    def ribbon(self, parent: ET.Element, *, route_id: str, points: list[list[float]],
               source: str, target: str, from_family: str, to_family: str,
               profile_name: str = "main_workflow") -> dict:
        profile = self.rules["arrow"]["profiles"][profile_name]
        widths = profile["widths_px"]
        opacities = profile["opacities"]
        sample_count = max(int(profile.get("minimum_samples", 48)), 56)
        body_end = 0.84

        def width_at(s: float) -> float:
            if s <= 0.5:
                return widths["tail"] + (widths["mid"] - widths["tail"]) * self._smoothstep(s / 0.5)
            if s <= body_end:
                return widths["mid"] + (widths["near_head"] - widths["mid"]) * self._smoothstep((s - 0.5) / (body_end - 0.5))
            return widths["near_head"] * (1 - self._smoothstep((s - body_end) / (1 - body_end)))

        left, right, samples = [], [], []
        for index in range(sample_count + 1):
            s = index / sample_count
            x, y = self._cubic(points, s)
            tx, ty = self._tangent(points, min(s, 0.99999))
            length = math.hypot(tx, ty) or 1.0
            nx, ny = -ty / length, tx / length
            width = width_at(s)
            left.append((x + nx * width / 2, y + ny * width / 2))
            right.append((x - nx * width / 2, y - ny * width / 2))
            samples.append({"s": s, "center": [x, y], "width": width})
        polygon = left + list(reversed(right))
        d = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in polygon) + " Z"

        defs = next((node for node in self.root if self._localname(node) == "defs"), None)
        if defs is None:
            defs = ET.Element(_tag("defs"))
            self.root.insert(0, defs)
        gradient_id = f"gradient-{route_id}"
        gradient = ET.SubElement(defs, _tag("linearGradient"), {
            "id": gradient_id, "gradientUnits": "userSpaceOnUse",
            "x1": str(points[0][0]), "y1": str(points[0][1]),
            "x2": str(points[-1][0]), "y2": str(points[-1][1]),
        })
        start_colors = self.rules["palette"]["families"][from_family]
        end_colors = self.rules["palette"]["families"][to_family]
        for offset, color, opacity in (
            ("0%", start_colors["dark"], opacities["tail"]),
            ("55%", start_colors["base"], opacities["mid"]),
            ("100%", end_colors["highlight"], opacities["head"]),
        ):
            ET.SubElement(gradient, _tag("stop"), {
                "offset": offset, "stop-color": color, "stop-opacity": str(opacity),
            })
        self.path(parent, d, fill=f"url(#{gradient_id})", element_id=route_id)
        return {
            "id": route_id, "type": profile_name, "source": source, "target": target,
            "control_points": points, "samples": samples,
            "widths_px": dict(widths), "opacities": dict(opacities),
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.root).write(target, encoding="utf-8", xml_declaration=True)
        return target
