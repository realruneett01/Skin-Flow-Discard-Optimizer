"""Loading of the YAML files under ``config/`` with provenance enforcement.

Every numeric parameter in a config file is a *leaf*: a mapping with ``value`` (or
``path``), ``status`` and ``SOURCE``. ``status`` is ``PLACEHOLDER`` or ``SOURCED``.
Loading fails if any leaf lacks provenance, so an unsourced constant can never slip
into a model or an ROI figure unmarked.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml

from skinflow_discard_optimizer.paths import CONFIG_DIR, REPO_ROOT

STATUSES = ("PLACEHOLDER", "SOURCED")


class ProvenanceError(ValueError):
    """A config leaf is missing its status or SOURCE."""


def _is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and ("value" in node or "path" in node)


def iter_leaves(tree: Any, prefix: str = "") -> Iterator[tuple[str, dict]]:
    """Yield ``(dotted_key, leaf)`` for every parameter leaf in a config tree."""
    if _is_leaf(tree):
        yield prefix, tree
    elif isinstance(tree, dict):
        for k, v in tree.items():
            yield from iter_leaves(v, f"{prefix}.{k}" if prefix else str(k))


def _is_bare_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_leaf_provenance(tree: dict, prefix: str) -> list[str]:
    issues = []
    if tree.get("status") not in STATUSES:
        issues.append(f"{prefix}: status must be one of {STATUSES}, got {tree.get('status')!r}")
    src = tree.get("SOURCE")
    if not isinstance(src, str) or not src.strip():
        issues.append(f"{prefix}: missing SOURCE")
    return issues


def _check_dict_node(tree: dict, prefix: str) -> list[str]:
    issues = []
    for k, v in tree.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if _is_bare_number(v):
            issues.append(f"{key}: bare number {v!r} without status/SOURCE")
        elif isinstance(v, list) and any(_is_bare_number(x) for x in v):
            issues.append(f"{key}: list of bare numbers without status/SOURCE")
        else:
            issues.extend(provenance_issues(v, key))
    return issues


def provenance_issues(tree: Any, prefix: str = "") -> list[str]:
    """List every leaf without valid provenance, and every bare number outside a leaf."""
    if _is_leaf(tree):
        return _check_leaf_provenance(tree, prefix)
    if isinstance(tree, dict):
        return _check_dict_node(tree, prefix)
    return []


def load_yaml(path: Path, *, enforce_provenance: bool = True) -> dict:
    with open(path, encoding="utf-8") as f:
        tree = yaml.safe_load(f) or {}
    if enforce_provenance:
        issues = provenance_issues(tree)
        if issues:
            raise ProvenanceError(f"{path.name}: " + "; ".join(issues))
    return tree


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """Load ``config/<name>.yaml`` with provenance enforced. Cached per process."""
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def value(tree: dict, dotted: str) -> Any:
    """Return the ``value`` of the leaf at ``dotted`` (e.g. ``"metal.billet_mass_kg"``)."""
    node: Any = tree
    for part in dotted.split("."):
        node = node[part]
    if not _is_leaf(node):
        raise KeyError(f"{dotted} is not a parameter leaf")
    return node["value"] if "value" in node else node["path"]


def placeholders(tree: dict) -> list[str]:
    """Dotted keys of every leaf still marked PLACEHOLDER."""
    return [k for k, leaf in iter_leaves(tree) if leaf.get("status") == "PLACEHOLDER"]


@dataclass(frozen=True)
class Economics:
    """Typed view of ``config/economics.yaml`` with the derived quantities the models use."""

    billet_price_per_kg: float
    remelt_credit_per_kg: float
    billet_mass_kg: float
    cycles_per_hour: float
    operating_hours_per_year: float
    throughput_value_per_second: float
    tariff_csv: Path
    tariff_fallback_per_kwh: float
    defect_loss_per_event: float
    defect_affected_length_m: float

    @property
    def cycles_per_year(self) -> float:
        return self.cycles_per_hour * self.operating_hours_per_year

    @property
    def net_metal_loss_per_kg(self) -> float:
        """Value lost per kg of extra discard: billet price minus what the remelt returns (c_m)."""
        return self.billet_price_per_kg - self.remelt_credit_per_kg

    @property
    def defect_loss_per_metre(self) -> float:
        """c_d in the decision formula, so that c_d * L_d equals the loss per defect event."""
        return self.defect_loss_per_event / self.defect_affected_length_m

    @classmethod
    def from_tree(cls, tree: dict) -> "Economics":
        v = lambda k: value(tree, k)  # noqa: E731
        return cls(
            billet_price_per_kg=float(v("metal.billet_price_per_kg")),
            remelt_credit_per_kg=float(v("metal.remelt_credit_per_kg")),
            billet_mass_kg=float(v("metal.billet_mass_kg")),
            cycles_per_hour=float(v("production.cycles_per_hour")),
            operating_hours_per_year=float(v("production.operating_hours_per_year")),
            throughput_value_per_second=float(v("throughput.value_per_second")),
            tariff_csv=(REPO_ROOT / v("energy.tariff_csv")).resolve(),
            tariff_fallback_per_kwh=float(v("energy.tariff_fallback_per_kwh")),
            defect_loss_per_event=float(v("defect.loss_per_event")),
            defect_affected_length_m=float(v("defect.affected_length_m")),
        )


def load_economics() -> Economics:
    return Economics.from_tree(load_config("economics"))
