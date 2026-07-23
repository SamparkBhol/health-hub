/**
 * Typed-state vocabulary shared by every surface.
 *
 * The API answers with machine-readable state codes rather than empty arrays with no
 * explanation. This module is the single place that turns one of those codes into
 * something a person can read. Nothing here invents a state: an unknown code is shown
 * verbatim with an honest "no description registered" body, never swallowed into a
 * blank panel and never softened into "no data found".
 */

export type Tone = "ok" | "warn" | "stop" | "sim" | "mute";

export interface TypedStateCopy {
  /** Short uppercase label for a chip. */
  label: string;
  /** Headline for a full empty state. */
  title: string;
  /** One or two sentences explaining what the state does and does not establish. */
  body: string;
  tone: Tone;
}

/**
 * Registered typed states. Keys are the exact `state.code` / `coverage_state` /
 * `answer_state` strings the API emits.
 */
const STATE_COPY: Record<string, TypedStateCopy> = {
  not_implemented_phase_one: {
    label: "Not implemented · phase one",
    title: "This capability is out of scope for phase one",
    body: "The route exists and reports its own status instead of returning a placeholder result. Nothing is estimated in its place.",
    tone: "mute",
  },
  insufficient_evidence: {
    label: "Insufficient evidence",
    title: "Not enough retrieved evidence to answer",
    body: "Fewer records were returned than this view needs. Insufficient evidence is a statement about collection, not about disease.",
    tone: "warn",
  },
  insufficient_training_data: {
    label: "Insufficient training data",
    title: "The target series cannot train a model",
    body: "The available history is a positive-only publication catalogue with no negative weeks and no denominators. No probability is produced from it.",
    tone: "stop",
  },
  awaiting_boundary_licence: {
    label: "Awaiting boundary licence",
    title: "Licensed geometry has not been cleared",
    body: "District or taluka geometry at this level is not licensed for redistribution yet. The map withholds the layer rather than drawing an unlicensed outline.",
    tone: "warn",
  },
  awaiting_sponsor_data: {
    label: "Awaiting sponsor data",
    title: "Authorised aggregates have not been supplied",
    body: "Routine surveillance aggregates, denominators and completeness metadata have not been provided to this deployment. Absence of a row is not zero disease.",
    tone: "warn",
  },
  target_series_ineligible: {
    label: "Target series ineligible",
    title: "No Odisha outbreak probability is available",
    body: "The historical target series fails the evidentiary gate for district-week forecasting. The forecast route refuses instead of returning a number.",
    tone: "stop",
  },
  public_catalogue_only_no_denominator: {
    label: "Catalogue only · no denominator",
    title: "Positive-only catalogue, no denominator",
    body: "Only published positive events are catalogued. There is no valid negative week, so a missing record can never be read as a zero.",
    tone: "warn",
  },
  simulation_only_not_odisha_risk: {
    label: "Simulation only",
    title: "Synthetic software harness",
    body: "These values come from a deterministic artificial panel used to verify the forecasting code path. They describe no Odisha district and no real disease.",
    tone: "sim",
  },
  dedup_cross_language_unvalidated: {
    label: "Dedup unvalidated",
    title: "Cross-language de-duplication is unvalidated",
    body: "The same event reported in Odia, Hindi and English may still be counted more than once. Counts are documents retrieved, not unique events.",
    tone: "warn",
  },
  community_demo_boundary: {
    label: "Community demo boundary",
    title: "Community boundary vintage in use",
    body: "Geometry is the DataMeet Census 2011 community release, not a survey-authoritative boundary. District identity is stable; edges are approximate.",
    tone: "mute",
  },
  fixture_fallback: {
    label: "Fixture fallback",
    title: "Bundled fixtures are answering this layer",
    body: "No live collection receipt backs these rows. Every value on screen is synthetic test data and carries a synthetic badge.",
    tone: "sim",
  },
  observed_for_registered_sources: {
    label: "Observed · registered sources",
    title: "Observed for registered sources only",
    body: "Records exist for the registered source set. Anything those sources never published is invisible here, in every district.",
    tone: "ok",
  },
  partial: {
    label: "Partial",
    title: "Partial coverage",
    body: "Some registered routes returned records and others did not. Treat the difference between districts as a difference in publishing, not in disease.",
    tone: "warn",
  },
  unknown: {
    label: "Unknown",
    title: "Coverage state not reported",
    body: "The response did not declare a coverage state. The interface will not assume one on its behalf.",
    tone: "mute",
  },
  not_observable_from_public_sources: {
    label: "Not observable",
    title: "Not observable from public sources",
    body: "The question asks for something public publishing cannot establish, such as incidence or burden.",
    tone: "stop",
  },
  ambiguous_scope: {
    label: "Ambiguous scope",
    title: "Scope could not be resolved",
    body: "The district or disease could not be pinned down well enough to answer. Narrow the question rather than accept a guess.",
    tone: "warn",
  },
  out_of_scope_clinical: {
    label: "Clinical advice refused",
    title: "Clinical advice is out of scope",
    body: "This system does not diagnose a person or recommend treatment under any phrasing.",
    tone: "stop",
  },
  audited_public_catalogue: {
    label: "Audited catalogue",
    title: "Answer comes from the frozen data audit",
    body: "The response summarises a hash-pinned audit of the public catalogue rather than live retrieval.",
    tone: "mute",
  },
  records_returned: {
    label: "Records returned",
    title: "Records returned",
    body: "Retrieved evidence is attached below with its provenance.",
    tone: "ok",
  },
  translation_unavailable_source_language_only: {
    label: "Source language only",
    title: "No machine translation is available here",
    body: "The translation service is not answering in this deployment, so evidence stays in the language it was published in. Nothing is translated in the browser as a substitute.",
    tone: "warn",
  },
  capacity_exceeded: {
    label: "Capacity exceeded",
    title: "The service refused the work rather than queue it",
    body: "CPU inference is serialised on this deployment. The request was declined so that a slow answer never looks like a stalled one; retry when the current job finishes.",
    tone: "warn",
  },
  native_odia_interface_not_validated: {
    label: "Odia interface unvalidated",
    title: "The Odia interface has not been validated",
    body: "Odia output is produced but has not been checked by a native reader against a test set. Treat it as a reading aid, not as a published translation.",
    tone: "warn",
  },
  language_review_required: {
    label: "Language review required",
    title: "A human language reviewer has to see this first",
    body: "Automatic handling abstained on this text — usually low OCR or extraction confidence — so it is held for review rather than guessed at.",
    tone: "warn",
  },
  no_current_authorised_outbreak_report_feed: {
    label: "No current authorised feed",
    title: "No current outbreak-report feed is authorised",
    body: "The catalogue this platform may use ends years before today, so no current-week probability is issued from it.",
    tone: "stop",
  },
};

export function humanizeCode(code: string): string {
  return code.replaceAll("_", " ").replace(/\s+/g, " ").trim().toUpperCase();
}

/** Never returns undefined. An unregistered code is surfaced, not hidden. */
export function typedState(code: string | null | undefined): TypedStateCopy {
  if (!code || !code.trim()) return STATE_COPY.unknown;
  const key = code.trim().toLowerCase();
  const registered = STATE_COPY[key];
  if (registered) return registered;
  return {
    label: humanizeCode(key),
    title: `Typed state: ${humanizeCode(key)}`,
    body: "The API returned this state code without a registered description. It is shown exactly as received rather than replaced with a generic empty message.",
    tone: "warn",
  };
}

export interface ReadDeferral {
  capability: string;
  code: string;
  reasonCode: string | null;
  startedAt: string | null;
  copy: TypedStateCopy;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/** Normalises the API's `deferrals[]` shape into something renderable. */
export function readDeferrals(raw: Array<Record<string, unknown>> | undefined): ReadDeferral[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((entry, index) => {
    const state = asRecord(entry.state);
    const code = asString(state.code) ?? asString(entry.code) ?? "unknown";
    return {
      capability: asString(entry.capability) ?? `deferral_${index + 1}`,
      code,
      reasonCode: asString(entry.reason_code),
      startedAt: asString(state.started_at),
      copy: typedState(code),
    };
  });
}

const SYNTHETIC_KEYS = [
  "is_synthetic",
  "isSynthetic",
  "is_synthetic_fixture",
  "is_fixture",
  "isFixture",
  "synthetic",
];

/**
 * True when the payload declares itself synthetic.
 *
 * Reads the canonical `is_synthetic` flag first, then the field names the current API
 * actually emits (`isFixture` on signals, `event.is_synthetic_fixture` on layer records),
 * so the badge keeps working if the backend consolidates on one field.
 */
export function isSynthetic(value: unknown): boolean {
  const record = asRecord(value);
  for (const key of SYNTHETIC_KEYS) {
    if (record[key] === true) return true;
  }
  const nested = asRecord(record.event);
  for (const key of SYNTHETIC_KEYS) {
    if (nested[key] === true) return true;
  }
  return false;
}

/** True when a whole response declares itself fixture-backed. */
export function isSyntheticResponse(fixtureMode: string | null | undefined, coverageState?: string | null): boolean {
  return fixtureMode === "fixture_only" || coverageState === "fixture_fallback";
}

export interface ReadWarning {
  code: string;
  severity: "info" | "warning" | "blocking";
  message: string;
}
