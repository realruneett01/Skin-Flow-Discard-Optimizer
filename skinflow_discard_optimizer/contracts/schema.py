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


class SignalContract(_Strict):
    version: int
    namespace_index: int
    signals: dict[str, SignalSpec]
    phase_aliases: dict[str, dict[str, str]]
    outputs: OutputNamespace

    @model_validator(mode="after")
    def _check_consistency(self):
        prefix = f"ns={self.namespace_index};s="
        all_tags = [s.tag for s in self.signals.values()] + [
            s.tag for s in self.outputs.signals.values()
        ]
        dupes = {t for t in all_tags if all_tags.count(t) > 1}
        if dupes:
            raise ValueError(f"duplicate OPC-UA tags: {sorted(dupes)}")
        for tag in all_tags:
            if not tag.startswith(prefix):
                raise ValueError(f"tag {tag} is not in namespace index {self.namespace_index}")

        ns = self.outputs.namespace
        for name, spec in self.outputs.signals.items():
            if not name.startswith(f"{ns}."):
                raise ValueError(f"output {name} is outside the {ns}.* namespace")
            if spec.tag != f"{prefix}{name}":
                raise ValueError(f"output {name} has tag {spec.tag}, expected {prefix}{name}")
        for name, spec in self.signals.items():
            if f";s={ns}." in spec.tag:
                raise ValueError(f"input signal {name} uses the reserved {ns}.* namespace")

        phases = set(self.signals["cycle_phase"].values or ())
        for source, mapping in self.phase_aliases.items():
            unknown = set(mapping.values()) - phases
            if unknown:
                raise ValueError(f"phase_aliases[{source}] maps to unknown phases {sorted(unknown)}")
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
