"""Canonical Matplotlib configuration for Active Learning manuscript figures."""

from __future__ import annotations

from ..builders.rule_loader import RuleBundle, load_rule_bundle


def build_rc(rules: RuleBundle | None = None) -> dict:
    """Translate canonical YAML tokens into Matplotlib rcParams."""
    bundle = rules or load_rule_bundle()
    fonts = bundle["fonts"]
    typography = bundle["typography"]["minimum_print_size_pt"]["journal_width_178mm"]
    svg = bundle["svg_style"]
    sans = [fonts["sans"]["preferred"], *fonts["sans"]["fallback"]]
    return {
        "font.family": "sans-serif",
        "font.sans-serif": sans,
        "mathtext.fontset": fonts["matplotlib"]["mathtext_fontset"],
        "svg.fonttype": fonts["matplotlib"]["svg_fonttype"],
        "pdf.fonttype": fonts["matplotlib"]["pdf_fonttype"],
        "ps.fonttype": fonts["matplotlib"]["ps_fonttype"],
        "axes.linewidth": svg["stroke"]["axis_pt"],
        "axes.labelsize": max(typography["axis_label"], 7.5),
        "axes.titlesize": max(typography["panel_title"], 8.5),
        "xtick.labelsize": max(typography["tick_label"], 6.8),
        "ytick.labelsize": max(typography["tick_label"], 6.8),
        "legend.fontsize": typography["body"],
        "lines.solid_capstyle": svg["stroke"]["linecap"],
        "lines.solid_joinstyle": svg["stroke"]["linejoin"],
        "figure.facecolor": bundle["palette"]["neutral"]["white"],
        "savefig.facecolor": bundle["palette"]["neutral"]["white"],
        "savefig.bbox": "tight",
    }


RC = build_rc()


def apply() -> None:
    """Apply the canonical project-wide Matplotlib settings."""
    from matplotlib import rcParams

    rcParams.update(RC)
