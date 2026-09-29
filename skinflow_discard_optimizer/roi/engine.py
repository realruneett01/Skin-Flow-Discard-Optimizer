"""Joint ROI calculation engine for the Press Value Platform (Task 6.2).

Combines economic returns across all three platform modules:
1. Module 1: Dead-Cycle Timer Optimizer (DCTO) - press throughput acceleration.
2. Module 2: Hydraulic Pump Energy Optimizer (HPEO) - electrical power reduction.
3. Module 3: Skin-Flow Discard Optimizer (SKDO) - metal yield recovery & discard minimization.

Enforces Rule 6: Always reports low, expected, and high intervals (95% CI)
driven by Phase 5 validation results, never a single uncalibrated number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

from skinflow_discard_optimizer.config import load_economics
from skinflow_discard_optimizer.paths import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# Baseline physical and performance quantities from Phase 5 evaluation
BASELINE_DISCARD_REDUCTION_MM_LOW: float = 9.790
BASELINE_DISCARD_REDUCTION_MM_EXP: float = 10.517
BASELINE_DISCARD_REDUCTION_MM_HIGH: float = 11.389

BASELINE_NET_SAVING_EUR_LOW: float = 0.3462
BASELINE_NET_SAVING_EUR_EXP: float = 0.3896
BASELINE_NET_SAVING_EUR_HIGH: float = 0.4633

BASELINE_DCTO_SECONDS_LOW: float = 1.20
BASELINE_DCTO_SECONDS_EXP: float = 1.80
BASELINE_DCTO_SECONDS_HIGH: float = 2.50

BASELINE_HPEO_KWH_LOW: float = 0.80
BASELINE_HPEO_KWH_EXP: float = 1.30
BASELINE_HPEO_KWH_HIGH: float = 1.80

DEFAULT_SPREAD_EUR_KG: float = 0.50
DEFAULT_TARIFF_EUR_KWH: float = 0.0643
DEFAULT_THROUGHPUT_EUR_S: float = 0.263889  # 950 EUR / 3600 s


@dataclass(frozen=True)
class ValueRange:
    """Represents a low/expected/high confidence interval under Rule 6."""

    low: float
    expected: float
    high: float
    unit: str

    def scale(self, factor: float) -> "ValueRange":
        return ValueRange(
            low=float(self.low * factor),
            expected=float(self.expected * factor),
            high=float(self.high * factor),
            unit=self.unit,
        )

    def add(self, other: "ValueRange") -> "ValueRange":
        return ValueRange(
            low=float(self.low + other.low),
            expected=float(self.expected + other.expected),
            high=float(self.high + other.high),
            unit=self.unit,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ROIParameters:
    """Plant-specific operational and economic inputs."""

    metal_price_eur_per_kg: float = 2.60
    remelt_credit_eur_per_kg: float = 2.10
    electricity_tariff_eur_mwh: float = 64.30
    press_rate_eur_per_hour: float = 950.0
    cycles_per_year: int = 309600
    defect_loss_per_event: float = 400.0
    billet_mass_kg: float = 95.0

    @property
    def metal_spread_eur_per_kg(self) -> float:
        return max(self.metal_price_eur_per_kg - self.remelt_credit_eur_per_kg, 0.0)

    @property
    def electricity_tariff_eur_kwh(self) -> float:
        return self.electricity_tariff_eur_mwh / 1000.0

    @property
    def throughput_value_eur_per_s(self) -> float:
        return self.press_rate_eur_per_hour / 3600.0

    @classmethod
    def load(cls) -> "ROIParameters":
        """Loads default parameters grounded in config/economics.yaml."""
        try:
            econ = load_economics()
            return cls(
                metal_price_eur_per_kg=float(econ.billet_price_per_kg),
                remelt_credit_eur_per_kg=float(econ.remelt_credit_per_kg),
                electricity_tariff_eur_mwh=float(econ.tariff_fallback_per_kwh * 1000.0),
                press_rate_eur_per_hour=float(econ.throughput_value_per_second * 3600.0),
                cycles_per_year=int(econ.cycles_per_hour * econ.operating_hours_per_year),
                defect_loss_per_event=float(econ.defect_loss_per_event),
                billet_mass_kg=float(econ.billet_mass_kg),
            )
        except (KeyError, ValueError, FileNotFoundError, AttributeError):
            return cls()


@dataclass(frozen=True)
class ModuleROI:
    """ROI assessment for an individual platform module."""

    module_name: str
    primary_lever: str
    physical_saving_per_billet: ValueRange
    saving_per_billet_eur: ValueRange
    annual_value_keur: ValueRange

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointROISummary:
    """Combined platform economic valuation across all three modules."""

    parameters: ROIParameters
    skdo: ModuleROI
    dcto: ModuleROI
    hpeo: ModuleROI
    total_saving_per_billet_eur: ValueRange
    total_annual_value_keur: ValueRange
    oracle_annual_ceiling_keur: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": asdict(self.parameters),
            "skdo": self.skdo.to_dict(),
            "dcto": self.dcto.to_dict(),
            "hpeo": self.hpeo.to_dict(),
            "total_saving_per_billet_eur": self.total_saving_per_billet_eur.to_dict(),
            "total_annual_value_keur": self.total_annual_value_keur.to_dict(),
            "oracle_annual_ceiling_keur": self.oracle_annual_ceiling_keur,
        }


def _annualize_saving(saving: ValueRange, cycles_per_year: int) -> ValueRange:
    scaled = saving.scale(cycles_per_year / 1000.0)
    return ValueRange(scaled.low, scaled.expected, scaled.high, "kEUR/year")


def _calc_skdo_roi(params: ROIParameters) -> ModuleROI:
    spread_ratio = params.metal_spread_eur_per_kg / DEFAULT_SPREAD_EUR_KG
    saving_billet = ValueRange(
        low=BASELINE_NET_SAVING_EUR_LOW * spread_ratio,
        expected=BASELINE_NET_SAVING_EUR_EXP * spread_ratio,
        high=BASELINE_NET_SAVING_EUR_HIGH * spread_ratio,
        unit="EUR/billet",
    )
    return ModuleROI(
        module_name="Skin-Flow Discard Optimizer (SKDO)",
        primary_lever="Metal Yield & Discard Minimization",
        physical_saving_per_billet=ValueRange(
            low=BASELINE_DISCARD_REDUCTION_MM_LOW,
            expected=BASELINE_DISCARD_REDUCTION_MM_EXP,
            high=BASELINE_DISCARD_REDUCTION_MM_HIGH,
            unit="mm discard",
        ),
        saving_per_billet_eur=saving_billet,
        annual_value_keur=_annualize_saving(saving_billet, params.cycles_per_year),
    )


def _calc_dcto_roi(params: ROIParameters) -> ModuleROI:
    v_s = params.throughput_value_eur_per_s
    saving_billet = ValueRange(
        low=BASELINE_DCTO_SECONDS_LOW * v_s,
        expected=BASELINE_DCTO_SECONDS_EXP * v_s,
        high=BASELINE_DCTO_SECONDS_HIGH * v_s,
        unit="EUR/billet",
    )
    return ModuleROI(
        module_name="Dead-Cycle Timer Optimizer (DCTO)",
        primary_lever="Dead-Cycle Throughput Compression",
        physical_saving_per_billet=ValueRange(
            low=BASELINE_DCTO_SECONDS_LOW,
            expected=BASELINE_DCTO_SECONDS_EXP,
            high=BASELINE_DCTO_SECONDS_HIGH,
            unit="seconds",
        ),
        saving_per_billet_eur=saving_billet,
        annual_value_keur=_annualize_saving(saving_billet, params.cycles_per_year),
    )


def _calc_hpeo_roi(params: ROIParameters) -> ModuleROI:
    tariff_kwh = params.electricity_tariff_eur_kwh
    saving_billet = ValueRange(
        low=BASELINE_HPEO_KWH_LOW * tariff_kwh,
        expected=BASELINE_HPEO_KWH_EXP * tariff_kwh,
        high=BASELINE_HPEO_KWH_HIGH * tariff_kwh,
        unit="EUR/billet",
    )
    return ModuleROI(
        module_name="Hydraulic Pump Energy Optimizer (HPEO)",
        primary_lever="Hydraulic Power Throttling & Unloading",
        physical_saving_per_billet=ValueRange(
            low=BASELINE_HPEO_KWH_LOW,
            expected=BASELINE_HPEO_KWH_EXP,
            high=BASELINE_HPEO_KWH_HIGH,
            unit="kWh",
        ),
        saving_per_billet_eur=saving_billet,
        annual_value_keur=_annualize_saving(saving_billet, params.cycles_per_year),
    )


class ROIEngine:
    """Evaluates multi-module economic returns and sensitivity ranges."""

    def __init__(self, default_params: ROIParameters | None = None) -> None:
        self.default_params = default_params or ROIParameters.load()

    def calculate_roi(self, parameters: ROIParameters | None = None) -> JointROISummary:
        """Calculates joint value across all three platform modules."""
        params = parameters or self.default_params

        skdo = _calc_skdo_roi(params)
        dcto = _calc_dcto_roi(params)
        hpeo = _calc_hpeo_roi(params)

        total_billet = skdo.saving_per_billet_eur.add(dcto.saving_per_billet_eur).add(hpeo.saving_per_billet_eur)
        total_annual = skdo.annual_value_keur.add(dcto.annual_value_keur).add(hpeo.annual_value_keur)

        # Oracle ceiling: perfect onset knowledge with 0 defect risk (~0.64 EUR/billet SKDO)
        oracle_billet = (0.640 * (params.metal_spread_eur_per_kg / DEFAULT_SPREAD_EUR_KG)) + dcto.saving_per_billet_eur.high + hpeo.saving_per_billet_eur.high
        oracle_annual = oracle_billet * params.cycles_per_year / 1000.0

        return JointROISummary(
            parameters=params,
            skdo=skdo,
            dcto=dcto,
            hpeo=hpeo,
            total_saving_per_billet_eur=total_billet,
            total_annual_value_keur=total_annual,
            oracle_annual_ceiling_keur=float(oracle_annual),
        )

    def evaluate_sensitivity(
        self,
        metal_spread_points: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
        tariff_points: tuple[float, ...] = (40.0, 60.0, 80.0, 100.0, 120.0),
        volume_points: tuple[int, ...] = (150000, 200000, 250000, 309600, 350000, 400000),
    ) -> dict[str, list[dict[str, Any]]]:
        """Generates parametric sensitivity tables across key economic levers."""
        base = self.default_params
        spread_results = []
        for s in metal_spread_points:
            p = ROIParameters(
                metal_price_eur_per_kg=base.remelt_credit_eur_per_kg + s,
                remelt_credit_eur_per_kg=base.remelt_credit_eur_per_kg,
                electricity_tariff_eur_mwh=base.electricity_tariff_eur_mwh,
                press_rate_eur_per_hour=base.press_rate_eur_per_hour,
                cycles_per_year=base.cycles_per_year,
            )
            res = self.calculate_roi(p)
            spread_results.append({
                "spread_eur_kg": s,
                "skdo_annual_low": res.skdo.annual_value_keur.low,
                "skdo_annual_exp": res.skdo.annual_value_keur.expected,
                "skdo_annual_high": res.skdo.annual_value_keur.high,
                "total_annual_exp": res.total_annual_value_keur.expected,
            })

        tariff_results = []
        for t in tariff_points:
            p = ROIParameters(
                metal_price_eur_per_kg=base.metal_price_eur_per_kg,
                remelt_credit_eur_per_kg=base.remelt_credit_eur_per_kg,
                electricity_tariff_eur_mwh=t,
                press_rate_eur_per_hour=base.press_rate_eur_per_hour,
                cycles_per_year=base.cycles_per_year,
            )
            res = self.calculate_roi(p)
            tariff_results.append({
                "tariff_eur_mwh": t,
                "hpeo_annual_low": res.hpeo.annual_value_keur.low,
                "hpeo_annual_exp": res.hpeo.annual_value_keur.expected,
                "hpeo_annual_high": res.hpeo.annual_value_keur.high,
                "total_annual_exp": res.total_annual_value_keur.expected,
            })

        volume_results = []
        for v in volume_points:
            p = ROIParameters(
                metal_price_eur_per_kg=base.metal_price_eur_per_kg,
                remelt_credit_eur_per_kg=base.remelt_credit_eur_per_kg,
                electricity_tariff_eur_mwh=base.electricity_tariff_eur_mwh,
                press_rate_eur_per_hour=base.press_rate_eur_per_hour,
                cycles_per_year=v,
            )
            res = self.calculate_roi(p)
            volume_results.append({
                "cycles_per_year": v,
                "total_annual_low": res.total_annual_value_keur.low,
                "total_annual_exp": res.total_annual_value_keur.expected,
                "total_annual_high": res.total_annual_value_keur.high,
            })

        return {
            "metal_spread_sensitivity": spread_results,
            "tariff_sensitivity": tariff_results,
            "volume_sensitivity": volume_results,
        }

    def save_roi_summary(self, path: Path | None = None) -> Path:
        """Saves default joint ROI valuation summary to JSON."""
        target = path or (ARTIFACTS_DIR / "roi_summary.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        summary = self.calculate_roi()
        with open(target, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info("Saved ROI summary to %s", target)
        return target
