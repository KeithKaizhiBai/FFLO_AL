#!/usr/bin/env python3
"""Figure-type-agnostic geometry primitives and spacing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def overlaps(self, other: "BBox", tolerance: float = 0.0) -> bool:
        return (
            self.x < other.right - tolerance
            and other.x < self.right - tolerance
            and self.y < other.bottom - tolerance
            and other.y < self.bottom - tolerance
        )

    def gap(self, other: "BBox") -> float:
        dx = max(self.x - other.right, other.x - self.right, 0.0)
        dy = max(self.y - other.bottom, other.y - self.bottom, 0.0)
        return hypot(dx, dy)

    def contains(self, other: "BBox", tolerance: float = 0.0) -> bool:
        return (
            other.x >= self.x - tolerance
            and other.y >= self.y - tolerance
            and other.right <= self.right + tolerance
            and other.bottom <= self.bottom + tolerance
        )

    @classmethod
    def from_mapping(cls, value: dict) -> "BBox":
        return cls(*(float(value[key]) for key in ("x", "y", "width", "height")))

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def snap(value: float, unit: float = 8.0) -> float:
    return round(value / unit) * unit


def pairwise(items: Iterable[tuple[str, BBox]]):
    values = list(items)
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            yield left, right


def visual_area_fraction(primary: Iterable[BBox], all_objects: Iterable[BBox]) -> float:
    primary_area = sum(box.area for box in primary)
    total = sum(box.area for box in all_objects)
    return primary_area / total if total else 0.0

