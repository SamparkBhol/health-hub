/**
 * Regression: the HMIS district-monthly panel was served by the API and appeared
 * nowhere in the interface. Its metric vocabulary, its period axis and — above
 * all — the sentence that keeps facility-reported records from being read as
 * incidence now live here.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  HMIS_EPISTEMIC_CAPTION, HMIS_METRICS, formatHmisPeriod, hmisMetricMeta, hmisPeriods, scaleHmisValue,
} from "../src/hmis.ts";

test("the caption denies incidence and de-duplication explicitly", () => {
  assert.match(HMIS_EPISTEMIC_CAPTION, /provisional/i);
  assert.match(HMIS_EPISTEMIC_CAPTION, /facility-reported/i);
  assert.match(HMIS_EPISTEMIC_CAPTION, /not deduplicated people/i);
  assert.match(HMIS_EPISTEMIC_CAPTION, /not population incidence/i);
});

test("every offered metric is one the API accepts", () => {
  // Mirrors the `allowed` set in services/api/public_health.py::hmis_map.
  const served = [
    "malaria_microscopy_positive_records",
    "malaria_microscopy_positivity",
    "malaria_positive_records",
    "malaria_test_positivity",
    "dengue_positive_records",
    "childhood_diarrhoea_records",
  ];
  for (const meta of HMIS_METRICS) assert.ok(served.includes(meta.value), meta.value);
  assert.equal(HMIS_METRICS.length, served.length);
});

test("metric labels and units are readable, never raw field names", () => {
  for (const meta of HMIS_METRICS) {
    assert.doesNotMatch(meta.label, /_/, meta.value);
    assert.doesNotMatch(meta.unit, /_/, meta.value);
    assert.match(meta.definition, /record|slide|test/i, meta.value);
  }
});

test("an unknown metric degrades without inventing a definition", () => {
  const meta = hmisMetricMeta("some_future_column");
  assert.equal(meta.label, "some future column");
  assert.match(meta.definition, /no definition/i);
});

test("periods are reconstructed newest first across the reported bounds", () => {
  const periods = hmisPeriods("2012-04-01", "2020-03-01");
  assert.equal(periods.length, 96);
  assert.equal(periods[0], "2020-03-01");
  assert.equal(periods.at(-1), "2012-04-01");
  assert.equal(new Set(periods).size, 96);
});

test("a year boundary is crossed correctly", () => {
  assert.deepEqual(
    hmisPeriods("2019-11-01", "2020-02-01"),
    ["2020-02-01", "2020-01-01", "2019-12-01", "2019-11-01"],
  );
});

test("missing or inverted bounds never invent a range", () => {
  assert.deepEqual(hmisPeriods(null, null), []);
  assert.deepEqual(hmisPeriods(undefined, "2020-03-01"), ["2020-03-01"]);
  assert.deepEqual(hmisPeriods("2020-03-01", "2019-01-01"), ["2019-01-01"]);
  assert.deepEqual(hmisPeriods("not-a-date", "also-not"), []);
});

test("a positivity ratio is drawn per 1,000 tests, not rounded away to zero", () => {
  const meta = hmisMetricMeta("malaria_test_positivity");
  assert.equal(meta.scale, 1000);
  // Balangir, March 2020: 5 positives over 21,784 tests.
  const drawn = scaleHmisValue(0.00022952625780389276, meta);
  assert.equal(drawn, 0.23);
  assert.notEqual(drawn?.toFixed(2), "0.00", "a real positive rate must not print as zero");
  assert.match(meta.unit, /per 1,000 tests/);
});

test("record counts are drawn exactly as the API sent them", () => {
  const meta = hmisMetricMeta("dengue_positive_records");
  assert.equal(meta.scale, 1);
  assert.equal(scaleHmisValue(26, meta), 26);
  assert.equal(scaleHmisValue(0, meta), 0, "a reported zero stays a zero");
});

test("an unreported value stays unmeasured and never becomes zero", () => {
  const meta = hmisMetricMeta("malaria_test_positivity");
  assert.equal(scaleHmisValue(null, meta), null);
  assert.equal(scaleHmisValue(undefined, meta), null);
  assert.equal(scaleHmisValue(Number.NaN, meta), null);
});

test("a period renders as a readable month", () => {
  assert.equal(formatHmisPeriod("2020-03-01"), "March 2020");
  assert.equal(formatHmisPeriod("2012-04-01"), "April 2012");
  assert.equal(formatHmisPeriod("garbage"), "garbage");
});
