/**
 * Regression: the Hindi starter prompt asked about malaria in Ganjam, and the
 * live API returns zero records for that pair — the first thing a Hindi-reading
 * operator saw was the agent finding nothing. Prompts are now filtered against
 * the evidence actually retained.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { EXAMPLE_CANDIDATES, availableExamples, conditionHolds } from "../src/examples.ts";

/** The shape of the district/disease pairs the live signal feed returns. */
const RETAINED = [
  { district: "Puri", disease: "covid_19" },
  { district: "Khordha", disease: "cancer" },
  { district: "Khordha", disease: "dengue" },
  { district: "District Unavailable", disease: "malaria" },
];

test("no offered prompt names a district and disease with no retained record", () => {
  for (const prompt of availableExamples(RETAINED)) {
    assert.ok(conditionHolds(RETAINED, prompt.needs), prompt.text);
  }
});

test("the Ganjam malaria prompt is not offered against this evidence set", () => {
  const offered = availableExamples(RETAINED).map((prompt) => prompt.text);
  assert.ok(!offered.some((text) => text.includes("गंजाम")), "Ganjam prompt must not be offered");
  assert.ok(!conditionHolds(RETAINED, { district: "Ganjam", disease: "malaria" }));
});

test("one prompt per slot, in slot order", () => {
  const offered = availableExamples(RETAINED);
  assert.deepEqual(offered.map((prompt) => prompt.slot), ["odia", "hindi", "english", "audit"]);
  assert.deepEqual(offered.map((prompt) => prompt.code), ["or", "hi", "en", "en"]);
});

test("a slot falls through to its next candidate when the first has no records", () => {
  const withoutKhordhaDengue = RETAINED.filter((row) => row.disease !== "dengue");
  const odia = availableExamples(withoutKhordhaDengue).find((prompt) => prompt.slot === "odia");
  assert.ok(odia);
  assert.ok(odia.text.includes("ପୁରୀ"), "should fall through to the Puri COVID-19 prompt");
});

test("a slot with no satisfiable candidate is dropped rather than shown broken", () => {
  const onlyCancer = [{ district: "Khordha", disease: "cancer" }];
  const offered = availableExamples(onlyCancer);
  assert.deepEqual(offered.map((prompt) => prompt.slot), ["hindi", "audit"]);
});

test("an empty evidence set still offers the prompts that need no evidence", () => {
  const offered = availableExamples([]);
  assert.deepEqual(offered.map((prompt) => prompt.slot), ["audit"]);
  assert.equal(offered[0].needs, null);
});

test("disease tags match regardless of spacing or case", () => {
  const rows = [{ district: "khordha", disease: "Dengue" }];
  assert.ok(conditionHolds(rows, { district: "Khordha", disease: "dengue" }));
  assert.ok(conditionHolds([{ district: "Puri", disease: "COVID 19" }], { disease: "covid_19" }));
});

test("every candidate declares either a condition or an explicit exemption", () => {
  for (const candidate of EXAMPLE_CANDIDATES) {
    assert.ok(candidate.needs !== undefined, candidate.text);
    assert.ok(candidate.gloss.length > 0, candidate.text);
  }
});
