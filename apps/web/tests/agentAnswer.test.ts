/**
 * Two regressions.
 *
 * 1. The turn header read "8 CITED RECORDS" whenever eight records were
 *    retrieved, even where the live API reported a single citation.
 * 2. Every evidence card printed the same `content_sha256` as its digest,
 *    because this deployment does not retain source content and every row
 *    carries the digest of the same non-retention placeholder.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { answerCounts, countsLabel, evidenceDigest } from "../src/agentAnswer.ts";
import type { AgentQueryResult } from "../src/types.ts";

/** Shaped from a real POST /api/v1/agent/query response. */
function result(overrides: Partial<AgentQueryResult> = {}): AgentQueryResult {
  return {
    intent: "evidence_search",
    answer_state: "records_returned",
    answer: "The district Puri has published dengue evidence [E1].",
    scope: { district_id: null, disease: "dengue", question_language: "en" },
    evidence: Array.from({ length: 8 }, (_unused, index) => ({ signal_id: `sig_${index + 1}` })),
    reason_codes: [],
    ...overrides,
  };
}

test("one citation over eight retrieved records is not reported as eight", () => {
  const counts = answerCounts(result({ citations: ["sig_1"] }));
  assert.equal(counts.retrieved, 8);
  assert.equal(counts.cited, 1);
  assert.equal(countsLabel(counts), "8 records retrieved · 1 record cited in the answer");
});

test("the citation list falls back to the generation trace", () => {
  const counts = answerCounts(result({
    generation: { cited_signal_ids: ["sig_2", "sig_3", "sig_2"] },
    retrieval: { considered: 27 },
  }));
  assert.equal(counts.cited, 2, "duplicate ids are one cited record");
  assert.equal(counts.considered, 27);
  assert.ok(counts.citedIds.has("sig_2"));
  assert.ok(!counts.citedIds.has("sig_1"));
});

test("an unreported citation list is stated, not assumed to equal retrieval", () => {
  const counts = answerCounts(result());
  assert.equal(counts.cited, null);
  assert.equal(counts.considered, null);
  assert.equal(countsLabel(counts), "8 records retrieved · citations not reported");
});

test("a deterministic answer that cites nothing says so", () => {
  const counts = answerCounts(result({ citations: [] }));
  assert.equal(counts.cited, 0);
  assert.equal(countsLabel(counts), "8 records retrieved · none cited in the answer text");
});

test("singular and empty retrievals read correctly", () => {
  const one = answerCounts(result({ evidence: [{ signal_id: "sig_1" }], citations: ["sig_1"] }));
  assert.equal(countsLabel(one), "1 record retrieved · 1 record cited in the answer");
  const none = answerCounts(result({ evidence: [], citations: [] }));
  assert.equal(countsLabel(none), "no records retrieved");
});

test("the snapshot digest is preferred over the constant placeholder digest", () => {
  const digest = evidenceDigest({
    content_sha256: "18b70b0352d343eeb297733e3c5a00c2f185d2789df3393b0674fd0ff096db8b",
    snapshot_content_sha256: "9f2c4ab1de7043aa55c1b0e6d2f7318844ce0a91ff23bb7c",
    source_snapshot_id: "snapshot_c983354fcc9903a5d7444877b5806c16",
  });
  assert.equal(digest.label, "Snapshot digest");
  assert.equal(digest.value, "9f2c4ab1de70…");
  assert.equal(digest.title, "9f2c4ab1de7043aa55c1b0e6d2f7318844ce0a91ff23bb7c");
});

test("without a snapshot digest the snapshot id is shown, labelled as an id", () => {
  const digest = evidenceDigest({
    content_sha256: "18b70b0352d343eeb297733e3c5a00c2f185d2789df3393b0674fd0ff096db8b",
    source_snapshot_id: "snapshot_c983354fcc9903a5d7444877b5806c16",
  });
  assert.equal(digest.label, "Snapshot id");
  assert.equal(digest.value, "c983354fcc99…");
  assert.doesNotMatch(digest.value, /^18b70b03/, "the placeholder digest must not be shown");
});

test("with neither field the card says the digest was not supplied", () => {
  const digest = evidenceDigest({ content_sha256: "18b70b0352d343ee" });
  assert.equal(digest.label, "Snapshot digest");
  assert.equal(digest.value, "not supplied");
  assert.equal(digest.title, null);
});

test("two live citations no longer share one digest string", () => {
  const first = evidenceDigest({
    content_sha256: "18b70b0352d343ee",
    source_snapshot_id: "snapshot_c983354fcc9903a5d7444877b5806c16",
  });
  const second = evidenceDigest({
    content_sha256: "18b70b0352d343ee",
    source_snapshot_id: "snapshot_689f5577c9b4388952cace381713c2ed",
  });
  assert.notEqual(first.value, second.value);
});
