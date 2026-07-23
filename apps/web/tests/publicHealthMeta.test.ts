/**
 * Regression: the "Observed disease burden" framing used to be a fixed string
 * above the NCVBDC map, so selecting ABER painted testing effort under a burden
 * heading and selecting Pf% painted parasite mix under it. The eyebrow is now
 * derived from the metric.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { MALARIA_METRICS, malariaMetricMeta } from "../src/publicHealthMeta.ts";

test("api, total_cases and deaths keep the burden framing", () => {
  for (const metric of ["api", "total_cases", "deaths"]) {
    const meta = malariaMetricMeta(metric);
    assert.equal(meta.framing, "burden", metric);
    assert.match(meta.eyebrow, /Observed disease burden/, metric);
  }
});

test("spr, aber and pf_percent are labelled effort and mix, not burden", () => {
  for (const metric of ["spr", "aber", "pf_percent"]) {
    const meta = malariaMetricMeta(metric);
    assert.equal(meta.framing, "effort_and_mix", metric);
    assert.doesNotMatch(meta.eyebrow, /Observed disease burden/, metric);
    assert.match(meta.eyebrow, /not burden/, metric);
  }
});

test("every selectable metric carries a readable label and unit", () => {
  for (const meta of MALARIA_METRICS) {
    assert.notEqual(meta.label, meta.value, meta.value);
    assert.doesNotMatch(meta.unit, /_/, `${meta.value} unit must not be a raw field name`);
    assert.ok(meta.caption.length > 40, meta.value);
  }
});

test("an unknown metric is never described as disease burden", () => {
  const meta = malariaMetricMeta("some_new_column");
  assert.equal(meta.framing, "effort_and_mix");
  assert.doesNotMatch(meta.eyebrow, /Observed disease burden/);
  assert.equal(meta.label, "some new column");
});
