"""Task 0.3: every number in a config file has a SOURCE or a PLACEHOLDER tag."""
import pytest

from skinflow_discard_optimizer.config import (
    ProvenanceError,
    iter_leaves,
    load_config,
    load_economics,
    load_yaml,
    placeholders,
    provenance_issues,
)
from skinflow_discard_optimizer.paths import CONFIG_DIR


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_config_file_has_full_provenance(path):
    tree = load_yaml(path, enforce_provenance=False)
    assert provenance_issues(tree) == []
    assert list(iter_leaves(tree)), f"{path.name} has no parameter leaves"


def test_economics_values_and_derived_quantities():
    econ = load_economics()
    assert econ.cycles_per_year == pytest.approx(43 * 7200)
    assert econ.net_metal_loss_per_kg == pytest.approx(0.50)
    assert econ.defect_loss_per_metre * econ.defect_affected_length_m == pytest.approx(
        econ.defect_loss_per_event
    )


def test_all_economics_are_currently_placeholders():
    tree = load_config("economics")
    assert set(placeholders(tree)) == {k for k, _ in iter_leaves(tree)}


@pytest.mark.parametrize("tree,needle", [
    ({"a": {"value": 1.0, "status": "PLACEHOLDER"}}, "missing SOURCE"),
    ({"a": {"value": 1.0, "SOURCE": "x"}}, "status must be"),
    ({"a": {"value": 1.0, "status": "GUESS", "SOURCE": "x"}}, "status must be"),
    ({"a": {"b": 3.5}}, "bare number"),
    ({"a": [1, 2]}, "bare numbers"),
])
def test_missing_provenance_is_caught(tree, needle, tmp_path):
    assert any(needle in i for i in provenance_issues(tree))
    p = tmp_path / "bad.yaml"
    import yaml
    p.write_text(yaml.safe_dump(tree))
    with pytest.raises(ProvenanceError):
        load_yaml(p)
