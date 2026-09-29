"""Typed view of ``contracts/signals.yaml``.

The YAML file is the single source of truth for signal names, OPC-UA tags, units,
rates and valid ranges. This module parses it into Pydantic models and checks the
file is internally consistent (unique tags, ranges ordered, outputs confined to
the ``Predictor`` namespace).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from skinflow_discard_optimizer.paths import SIGNALS_YAML

SignalType = Literal["float", "int", "bool", "str", "enum"]
Owner = Literal["dcto", "hpeo", "skinflow", "plant"]
Kind = Literal["measured", "demand", "derived", "context"]

TAG_PATTERN = r"^ns=\d+;s=[A-Za-z0-9_.]+$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _TypedSpec(_Strict):
    tag: str = Field(pattern=TAG_PATTERN)
    unit: str
    type: SignalType
    range: tuple[float, float] | None = None
    values: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _check_type_fields(self):
        if self.type in ("float", "int"):
            if self.range is None:
                raise ValueError(f"{self.tag}: numeric signal needs a range")
            lo, hi = self.range
            if not lo < hi:
                raise ValueError(f"{self.tag}: range must satisfy min < max, got {self.range}")
        if self.type == "enum" and not self.values:
            raise ValueError(f"{self.tag}: enum signal needs 'values'")
        if self.type != "enum" and self.values is not None:
            raise ValueError(f"{self.tag}: only enum signals may define 'values'")
        return self


class SignalSpec(_TypedSpec):
    rate_hz: float = Field(ge=0)
    owner: Owner
    kind: Kind
    aliases: dict[str, str] = Field(default_factory=dict)


class OutputSpec(_TypedSpec):
    description: str


class OutputNamespace(_Strict):
    namespace: str
    signals: dict[str, OutputSpec]


def _validate_tags(signals: dict, output_signals: dict, prefix: str, namespace_index: int) -> None:
    all_tags = [s.tag for s in signals.values()] + [s.tag for s in output_signals.values()]
    dupes = {t for t in all_tags if all_tags.count(t) > 1}
    if dupes:
        raise ValueError(f"duplicate OPC-UA tags: {sorted(dupes)}")
    for tag in all_tags:
        if not tag.startswith(prefix):
            raise ValueError(f"tag {tag} is not in namespace index {namespace_index}")


def _validate_output_namespaces(signals: dict, outputs: OutputNamespace, prefix: str) -> None:
    ns = outputs.namespace
    for name, spec in outputs.signals.items():
        if not name.startswith(f"{ns}."):
            raise ValueError(f"output {name} is outside the {ns}.* namespace")
        if spec.tag != f"{prefix}{name}":
            raise ValueError(f"output {name} has tag {spec.tag}, expected {prefix}{name}")
    for name, spec in signals.items():
        if f";s={ns}." in spec.tag:
            raise ValueError(f"input signal {name} uses the reserved {ns}.* namespace")


def _validate_phase_aliases(signals: dict, phase_aliases: dict) -> None:
    phases = set(signals["cycle_phase"].values or ())
    for source, mapping in phase_aliases.items():
        unknown = set(mapping.values()) - phases
        if unknown:
            raise ValueError(f"phase_aliases[{source}] maps to unknown phases {sorted(unknown)}")


class SignalContract(_Strict):
    version: int
    namespace_index: int
    signals: dict[str, SignalSpec]
    phase_aliases: dict[str, dict[str, str]]
    outputs: OutputNamespace

    @model_validator(mode="after")
    def _check_consistency(self):
        prefix = f"ns={self.namespace_index};s="
        _validate_tags(self.signals, self.outputs.signals, prefix, self.namespace_index)
        _validate_output_namespaces(self.signals, self.outputs, prefix)
        _validate_phase_aliases(self.signals, self.phase_aliases)
        return self

    def canonical_phase(self, name: str, source: str) -> str:
        """Map a phase name as spelled by ``source`` (dcto or hpeo) to the canonical name."""
        try:
            return self.phase_aliases[source][name]
        except KeyError as exc:
            raise KeyError(f"unknown phase {name!r} for source {source!r}") from exc


def load_contract_file(path: Path) -> SignalContract:
    with open(path, encoding="utf-8") as f:
        return SignalContract.model_validate(yaml.safe_load(f))


@lru_cache(maxsize=1)
def load_contract() -> SignalContract:
    """The repo's contract, parsed once per process."""
    return load_contract_file(SIGNALS_YAML)
