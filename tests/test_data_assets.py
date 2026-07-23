import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_boundary_asset_is_pinned_and_has_thirty_unique_districts() -> None:
    directory = ROOT / "data" / "boundaries"
    manifest = json.loads((directory / "manifest.json").read_text())
    asset = directory / manifest["asset"]
    collection = json.loads(asset.read_text())
    names = [item["properties"]["canonical_name"] for item in collection["features"]]
    identifiers = [item["properties"]["district_id"] for item in collection["features"]]
    assert manifest["licence"] == "CC-BY-2.5-IN"
    assert manifest["feature_count"] == 30
    assert len(names) == len(set(names)) == 30
    assert len(identifiers) == len(set(identifiers)) == 30
    assert _sha256(asset) == manifest["asset_sha256"]


def test_epiclim_audit_contains_reproduced_findings_and_refusal() -> None:
    audit = json.loads((ROOT / "data" / "epiclim" / "audit.json").read_text())
    assert audit["source"]["sha256"] == (
        "7348076420202f8146ec2d36f36423cebd31af3cfbb8784e8c01e84b8ce0fb31"
    )
    assert audit["national"]["rows"] == 8_985
    assert audit["national"]["week_index_mismatch_gt_one_week_rows"] == 2_517
    assert audit["odisha"]["rows"] == 358
    assert audit["odisha"]["disease_counts"]["Dengue"] == 2
    assert audit["eligibility"]["district_week_count_forecast"] == "ineligible"
    assert audit["eligibility"]["missing_rows_are_zero"] is False
