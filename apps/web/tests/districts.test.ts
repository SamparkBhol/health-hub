/**
 * Regression: an unresolved place must never be rendered as a district row.
 *
 * The published signal set really does carry records tagged "District
 * Unavailable". Before this fix they were sorted into the district x disease
 * grid as though a district of that name existed — 22 records deep, near the top
 * of the row order, next to Puri and Khordha.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { normalizeName, resolveDistrictId, tallyDistrictDisease } from "../src/districts.ts";

const DISTRICTS = [
  { id: "OD-DIST-puri", name: "Puri" },
  { id: "OD-DIST-khordha", name: "Khordha" },
  { id: "OD-DIST-ganjam", name: "Ganjam" },
];

test("a census alias still resolves onto the canonical district", () => {
  assert.equal(normalizeName("Khurda"), "khordha");
  assert.equal(resolveDistrictId("Khurda", DISTRICTS), "OD-DIST-khordha");
  assert.equal(resolveDistrictId("  puri ", DISTRICTS), "OD-DIST-puri");
});

test("an unresolvable place name resolves to null, not to a made-up id", () => {
  assert.equal(resolveDistrictId("District Unavailable", DISTRICTS), null);
  assert.equal(resolveDistrictId("", DISTRICTS), null);
  assert.equal(resolveDistrictId(null, DISTRICTS), null);
});

test("District Unavailable is not a row and is counted separately", () => {
  const signals = [
    { district: "Puri", disease: "covid_19" },
    { district: "Puri", disease: "covid_19" },
    { district: "Khordha", disease: "cancer" },
    { district: "District Unavailable", disease: "malaria" },
    { district: "District Unavailable", disease: "malaria" },
    { district: "District Unavailable", disease: "dengue" },
  ];
  const tally = tallyDistrictDisease(signals, DISTRICTS);

  assert.deepEqual(tally.rows.map((row) => row.districtId), ["OD-DIST-puri", "OD-DIST-khordha"]);
  for (const row of tally.rows) {
    assert.ok(DISTRICTS.some((district) => district.id === row.districtId));
  }
  assert.equal(tally.unresolved.records, 3);
  assert.deepEqual(tally.unresolved.names, [{ name: "District Unavailable", count: 3 }]);
  assert.deepEqual(
    tally.unresolved.diseases,
    [{ name: "malaria", count: 2 }, { name: "dengue", count: 1 }],
  );
});

test("column totals equal the sum of the visible cells", () => {
  const signals = [
    { district: "Puri", disease: "covid_19" },
    { district: "Khordha", disease: "covid_19" },
    { district: "District Unavailable", disease: "covid_19" },
  ];
  const tally = tallyDistrictDisease(signals, DISTRICTS);
  const covid = tally.columns.find((column) => column.disease === "covid_19");

  assert.equal(covid?.total, 2, "the unresolved record must not inflate the disease column");
  const cellSum = [...tally.cells.entries()]
    .filter(([cell]) => cell.endsWith("|covid_19"))
    .reduce((total, [, count]) => total + count, 0);
  assert.equal(cellSum, covid?.total);
});

test("a disease seen only on unresolved records produces no column", () => {
  const tally = tallyDistrictDisease(
    [{ district: "District Unavailable", disease: "filariasis" }],
    DISTRICTS,
  );
  assert.deepEqual(tally.rows, []);
  assert.deepEqual(tally.columns, []);
  assert.equal(tally.unresolved.records, 1);
});

test("silent districts are counted without going negative", () => {
  const tally = tallyDistrictDisease(
    [{ district: "District Unavailable", disease: "malaria" }],
    DISTRICTS,
  );
  assert.equal(DISTRICTS.length - tally.rows.length, 3);
});
