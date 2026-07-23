/**
 * Reading an agent answer honestly: how many records were retrieved, how many
 * the model actually cited, and which digest identifies a citation.
 *
 * Two traps live here.
 *
 * 1. Retrieved is not cited. The client asks for up to eight evidence records
 *    and the API returns the whole retrieved set, but the written answer may
 *    reference one of them. Labelling the retrieved count "cited records"
 *    inflates the apparent grounding of the answer by up to eight times, so the
 *    two counts are reported separately and an unreported citation list is
 *    stated as unreported rather than assumed to equal the retrieval.
 *
 * 2. `content_sha256` is the same value on every live citation: this deployment
 *    does not retain source content, so every row carries the digest of the
 *    same non-retention placeholder. Printing it as "digest" implies a
 *    per-record fingerprint that does not exist. The snapshot digest — genuinely
 *    distinct per fetch — is preferred, and when the response does not carry one
 *    the snapshot identifier is labelled as an identifier.
 *
 * Deliberately free of runtime imports so it can be unit-tested on its own.
 */

import type { AgentEvidenceCitation, AgentQueryResult } from "./types";

export interface AnswerCounts {
  /** Evidence records the API attached to the answer. */
  retrieved: number;
  /** Records the answer text actually cites; `null` when the API did not report. */
  cited: number | null;
  /** Records the retriever scored before truncation; `null` when not reported. */
  considered: number | null;
  /** Signal ids the answer cites, for marking the matching evidence cards. */
  citedIds: Set<string>;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

export function answerCounts(result: AgentQueryResult): AnswerCounts {
  const evidence = Array.isArray(result.evidence) ? result.evidence : [];
  const reported = Array.isArray(result.citations)
    ? stringList(result.citations)
    : Array.isArray(result.generation?.cited_signal_ids)
      ? stringList(result.generation?.cited_signal_ids)
      : null;
  const considered = typeof result.retrieval?.considered === "number" ? result.retrieval.considered : null;
  return {
    retrieved: evidence.length,
    cited: reported === null ? null : new Set(reported).size,
    considered,
    citedIds: new Set(reported ?? []),
  };
}

function plural(count: number, singular: string, many: string): string {
  return `${count} ${count === 1 ? singular : many}`;
}

/** The line rendered beside the answer state. Never calls a retrieval a citation. */
export function countsLabel(counts: AnswerCounts): string {
  const retrieved = `${plural(counts.retrieved, "record", "records")} retrieved`;
  if (counts.cited === null) return `${retrieved} · citations not reported`;
  if (counts.retrieved === 0) return "no records retrieved";
  if (counts.cited === 0) return `${retrieved} · none cited in the answer text`;
  return `${retrieved} · ${plural(counts.cited, "record", "records")} cited in the answer`;
}

export interface EvidenceDigest {
  label: string;
  value: string;
  /** Full value for a tooltip, when the displayed value is truncated. */
  title: string | null;
}

function trimmed(value: string | null | undefined): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function shorten(value: string, length = 12): string {
  return value.length > length ? `${value.slice(0, length)}…` : value;
}

/**
 * The one identifier worth printing on an evidence card.
 *
 * `snapshot_content_sha256` is the digest of what was actually fetched, so it
 * differs per fetch and is worth showing. Its absence is reported as an
 * identifier instead — never as the constant placeholder digest.
 */
export function evidenceDigest(citation: AgentEvidenceCitation): EvidenceDigest {
  const digest = trimmed(citation.snapshot_content_sha256);
  if (digest) return { label: "Snapshot digest", value: shorten(digest), title: digest };
  const snapshot = trimmed(citation.source_snapshot_id);
  if (snapshot) {
    const bare = snapshot.replace(/^snapshot_/i, "");
    return { label: "Snapshot id", value: shorten(bare), title: snapshot };
  }
  return { label: "Snapshot digest", value: "not supplied", title: null };
}
