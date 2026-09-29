"""Pydantic models generated from the signal contract, and validators built on them.

Two paths share the same ranges:

* ``validate_value`` / ``validate_record`` go through generated Pydantic models and
  are used for per-cycle scalars and for outputs.
* ``check_array`` is a vectorised check for 1 kHz streams, where building one
  Pydantic object per sample would cost more than the physics model itself.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from skinflow_discard_optimizer.contracts.schema import (
    OutputSpec,
    SignalContract,
    SignalSpec,
    load_contract,
)


class ContractViolation(ValueError):
    """Raised when data breaks the signal contract."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


class AdvisoryViolation(PermissionError):
    """Raised when code tries to publish outside the ``Predictor.*`` namespace."""


def _value_field(spec: SignalSpec | OutputSpec, **extra: Any) -> tuple[Any, Any]:
    if spec.type == "float":
        lo, hi = spec.range
        return float, Field(ge=lo, le=hi, allow_inf_nan=False, **extra)
    if spec.type == "int":
        lo, hi = spec.range
        return int, Field(ge=int(lo), le=int(hi), **extra)
    if spec.type == "bool":
        return bool, Field(**extra)
    if spec.type == "enum":
        return Literal[tuple(spec.values)], Field(**extra)
    return str, Field(min_length=1, **extra)


def _plain(value: Any) -> Any:
    """Unwrap numpy scalars so strict validation treats them like Python scalars."""
    return value.item() if isinstance(value, np.generic) else value


def _model_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace(".", "_").split("_")) + "Value"


@lru_cache(maxsize=None)
def value_model(name: str) -> type[BaseModel]:
    """Generated Pydantic model with a single ``value`` field for one input or output signal."""
    contract = load_contract()
    spec = contract.signals.get(name) or contract.outputs.signals.get(name)
    if spec is None:
        raise KeyError(f"signal {name!r} is not in the contract")
    typ, field = _value_field(spec)
    return create_model(
        _model_name(name),
        __config__=ConfigDict(strict=True, extra="forbid"),
        value=(typ, field),
    )


@lru_cache(maxsize=1)
def record_model() -> type[BaseModel]:
    """Generated model covering every event-rate (per-cycle) input signal. All fields optional."""
    contract = load_contract()
    fields = {}
    for name, spec in contract.signals.items():
        typ, field = _value_field(spec, default=None)
        fields[name] = (typ | None, field)
    return create_model(
        "CycleRecord", __config__=ConfigDict(strict=True, extra="forbid"), **fields
    )


def _issues(err: ValidationError, prefix: str = "") -> list[str]:
    out = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"] if p != "value")
        where = f"{prefix}{loc}" if loc else prefix.rstrip(".")
        out.append(f"{where}: {e['msg']} (got {e.get('input')!r})")
    return out


def validate_value(name: str, value: Any) -> Any:
    """Validate one value of signal ``name``; returns the coerced value or raises ContractViolation."""
    try:
        return value_model(name)(value=_plain(value)).value
    except ValidationError as err:
        raise ContractViolation(_issues(err, prefix=name)) from None


def validate_record(record: Mapping[str, Any]) -> BaseModel:
    """Validate a dict of per-cycle signals. Unknown keys and out-of-range values are rejected."""
    try:
        return record_model()(**{k: _plain(v) for k, v in record.items()})
    except ValidationError as err:
        raise ContractViolation(_issues(err)) from None


def check_array(name: str, values: np.ndarray) -> np.ndarray:
    """Vectorised range check for a high-rate numeric stream.

    Returns a boolean mask, True where the sample is valid (finite and in range).
    """
    spec = load_contract().signals[name]
    if spec.type not in ("float", "int"):
        raise TypeError(f"{name} is not numeric")
    lo, hi = spec.range
    v = np.asarray(values, dtype=float)
    return np.isfinite(v) & (v >= lo) & (v <= hi)


def validate_array(name: str, values: np.ndarray) -> np.ndarray:
    """Strict version of ``check_array``: raises if any sample is invalid."""
    mask = check_array(name, values)
    if not mask.all():
        bad = np.flatnonzero(~mask)
        v = np.asarray(values, dtype=float)
        shown = ", ".join(f"[{i}]={v[i]!r}" for i in bad[:5])
        raise ContractViolation([f"{name}: {bad.size} of {mask.size} samples invalid ({shown})"])
    return np.asarray(values, dtype=float)


def ensure_advisory(output_name: str, contract: SignalContract | None = None) -> str:
    """Return the OPC-UA tag for an advisory output, or raise if it is not a Predictor.* output.

    Every publish path must call this. It is the code-level form of the rule that the
    module never writes shear, valve or pump setpoints.
    """
    contract = contract or load_contract()
    ns = contract.outputs.namespace
    spec = contract.outputs.signals.get(output_name)
    if spec is None or not output_name.startswith(f"{ns}."):
        raise AdvisoryViolation(
            f"refusing to publish {output_name!r}: only {ns}.* advisory outputs are allowed"
        )
    return spec.tag


def validate_output(output_name: str, value: Any) -> tuple[str, Any]:
    """Check an advisory output name and value. Returns ``(opcua_tag, value)``."""
    tag = ensure_advisory(output_name)
    return tag, validate_value(output_name, value)
