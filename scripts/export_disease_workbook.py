#!/usr/bin/env python3
"""Export the historical disease and environment panels as one Excel workbook.

Everything the predictive model reads, in a form a domain reviewer can open and
check by hand. Written with the standard library only -- xlsx is a zip of XML,
so this needs no third-party dependency and cannot drift from the runtime.

    uv run python scripts/export_disease_workbook.py

Sheets:
    ncvbdc_annual        official NCVBDC district-year malaria, 2010-2024
    hmis_monthly         Odisha HMIS district-month indicators
    model_rows           the exact rows the model trains on, features included
    model_ladder         rolling-origin scores for every competitor
    outlook_current      the current three-month district outlook
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "public_health" / "odisha_disease_history.xlsx"


def _cell(value: object) -> str:
    """One xlsx cell. Numbers stay numeric so a reader can chart them directly."""

    if value is None or value == "":
        return "<c/>"
    if isinstance(value, bool):
        return f"<c t='str'><is><t>{value}</t></is></c>"
    if isinstance(value, int | float):
        return f"<c><v>{value}</v></c>"
    text = str(value)
    try:
        return f"<c><v>{float(text)}</v></c>"
    except ValueError:
        return f"<c t='inlineStr'><is><t>{escape(text)}</t></is></c>"


def _sheet_xml(rows: list[list[object]]) -> str:
    body = "".join(
        "<row>" + "".join(_cell(value) for value in row) + "</row>" for row in rows
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{body}</sheetData></worksheet>"
    )


def write_workbook(path: Path, sheets: dict[str, list[list[object]]]) -> None:
    names = list(sheets)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.'
            "openxmlformats-package.relationships+xml\"/>"
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.'
            'openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{index + 1}.xml"'
                ' ContentType="application/vnd.openxmlformats-officedocument.'
                'spreadsheetml.worksheet+xml"/>'
                for index in range(len(names))
            )
            + "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="xl/workbook.xml"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
            ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(
                f'<sheet name="{escape(name)}" sheetId="{index + 1}" r:id="rId{index + 1}"/>'
                for index, name in enumerate(names)
            )
            + "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{index + 1}" Target="worksheets/sheet{index + 1}.xml"'
                ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
                for index in range(len(names))
            )
            + "</Relationships>",
        )
        for index, name in enumerate(names):
            archive.writestr(
                f"xl/worksheets/sheet{index + 1}.xml", _sheet_xml(sheets[name])
            )


def _csv_sheet(path: Path) -> list[list[object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [list(row) for row in csv.reader(handle)]


def main() -> int:
    from packages.forecasting.public_hmis import FEATURE_NAMES, build_model_rows, load_model
    from services.api.public_health import public_outlook_map

    sheets: dict[str, list[list[object]]] = {}

    sheets["ncvbdc_annual"] = _csv_sheet(
        ROOT / "data" / "public_health" / "ncvbdc_malaria_annual.csv"
    )
    sheets["hmis_monthly"] = _csv_sheet(
        ROOT / "data" / "public_health" / "hmis_district_monthly.csv"
    )

    rows = build_model_rows()
    header: list[object] = [
        "district_id",
        "period_start",
        "positivity",
        "threshold_p75_trailing",
        "target_elevated",
        *FEATURE_NAMES,
    ]
    sheets["model_rows"] = [header] + [
        [
            row.district_id,
            row.period_start.isoformat(),
            row.positivity,
            row.threshold,
            row.target,
            *row.features,
        ]
        for row in rows
    ]

    model = load_model() or {}
    pooled = model.get("pooled", {})
    sheets["model_ladder"] = [
        ["model", "brier", "log_score", "auc", "expected_calibration_error"],
        *[
            [
                name,
                scores.get("brier"),
                scores.get("log_score"),
                scores.get("auc"),
                scores.get("expected_calibration_error"),
            ]
            for name, scores in sorted(
                pooled.items(), key=lambda item: item[1].get("brier", 1.0)
            )
        ],
        [],
        ["selected_by_brier", model.get("selected_by_brier")],
        ["beats_unconditional_climatology", model.get("beats_unconditional_climatology")],
        [
            "brier_skill_score_vs_unconditional",
            model.get("brier_skill_score_vs_unconditional"),
        ],
        ["modeling_rows", model.get("modeling_rows")],
        ["events", model.get("events")],
    ]

    outlook = public_outlook_map(horizon_month=1)
    records = outlook["records"]
    columns = [
        "district_id",
        "district_name",
        "research_indicator_probability",
        "surveillance_priority_score",
        "forecast_precipitation_mean_mm",
        "forecast_precipitation_p10_mm",
        "forecast_precipitation_p90_mm",
        "forecast_temperature_mean_c",
    ]
    sheets["outlook_current"] = [columns] + [
        [record.get(column) for column in columns] for record in records
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_workbook(OUTPUT, sheets)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "path": str(OUTPUT.relative_to(ROOT)),
        "sheets": {name: len(rows) - 1 for name, rows in sheets.items()},
        "bytes": OUTPUT.stat().st_size,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
