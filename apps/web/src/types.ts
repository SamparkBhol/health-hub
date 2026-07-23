import type { AnswerLanguage } from "./languages";

export type LayerType =
  | "public_source_signal"
  | "verified_event"
  | "official_event_catalogue"
  | "observed_surveillance"
  | "forecast"
  | "coverage";

export interface Warning {
  code: string;
  severity: "info" | "warning" | "blocking";
  message: string;
}

export interface Envelope<T> {
  schema_version: string;
  request_id: string;
  generated_at: string;
  deployment_profile: string;
  context: {
    layer_type: LayerType | "environment" | "not_applicable";
    coverage_state: string;
    [key: string]: unknown;
  };
  provenance: Array<Record<string, unknown>>;
  warnings: Warning[];
  deferrals: Array<Record<string, unknown>>;
  data: T;
}

export interface Signal {
  id: string;
  title: string;
  titleOriginal?: string;
  language: "or" | "hi" | "en" | "mixed" | "und";
  source: string;
  district: string;
  disease: string;
  assertion: "affirmed" | "not_affirmed" | "speculative" | "non_current";
  reviewState: "unreviewed" | "verified" | "rejected";
  retrievedAt: string;
  evidence: string;
  hash?: string;
  isFixture?: boolean;
  sourceSnapshotId?: string;
  accessPath?: string | null;
  canonicalUrl?: string | null;
}

export type SignalLanguage = Signal["language"];
export type SignalAssertion = Signal["assertion"];

export interface SignalFilters {
  disease?: string;
  language?: SignalLanguage;
  assertion?: SignalAssertion | "all";
  retrieved_from?: string;
  retrieved_to?: string;
}

export interface PublishedSignalDistrictAggregate {
  district_id: string;
  published_signal_count: number;
  first_retrieved_at: string;
  last_retrieved_at: string;
}

export interface PublishedSignalMap {
  metric: "published_signal_count";
  time_axis: "retrieval_time_not_event_onset";
  fixture_mode: "live_only" | "fixture_only" | "all";
  filters: {
    disease: string | null;
    language: SignalLanguage | null;
    assertion: SignalAssertion | null;
    retrieved_from: string | null;
    retrieved_to: string | null;
  };
  districts: PublishedSignalDistrictAggregate[];
}

export interface PublicMalariaMap {
  status: "observed_public_data";
  disease: "malaria";
  year: number;
  available_years: number[];
  metric: string;
  metric_definition: string;
  geography: string;
  source_scope: string;
  records: Array<{
    district_id: string;
    district_name: string;
    year: number;
    value: number | null;
    api: number;
    total_cases: number | null;
    population_thousands: number | null;
    aber: number;
    spr: number;
    pf_percent: number;
    deaths: number | null;
    observation_state: string;
    source_url: string;
    source_sha256: string;
  }>;
}

/**
 * The district-monthly HMIS panel.
 *
 * Every numeric field is a count of facility-submitted service or test records
 * for that month — provisional, revised by later submissions, and carrying no
 * population denominator. They are not deduplicated people and not incidence.
 */
export interface PublicHmisMap {
  status: "observed_public_data";
  /** Month-start of the selected period, e.g. `2020-03-01`. */
  period: string;
  available_period_start: string;
  available_period_end: string;
  metric: string;
  /** The API's own scope sentence for the metric. */
  metric_scope: string;
  is_synthetic?: boolean;
  records: Array<{
    district_id: string;
    district_name: string;
    value: number | null;
    malaria_microscopy_positive_records: number | null;
    malaria_microscopy_tests: number | null;
    malaria_microscopy_positivity: number | null;
    malaria_positive_records: number | null;
    malaria_tests: number | null;
    malaria_test_positivity: number | null;
    dengue_positive_records: number | null;
    childhood_diarrhoea_records: number | null;
    observation_state: string;
    source_url: string;
    resource_url?: string | null;
    source_sha256: string;
  }>;
}

export interface PublicOutlook {
  status: "research_outlook";
  disease: "malaria";
  horizon_month: number;
  forecast_target: string;
  not_target: string;
  selected_model: string;
  environment_promoted: boolean;
  environment_in_served_vector: boolean;
  environment_ablation_result: string;
  skill_attribution: string;
  model_feature_names: string[];
  model_evaluation: Record<string, Record<string, unknown>>;
  forecast_calibration_state: string;
  forecast_error_note: string;
  environment_provider: string;
  environment_model: string;
  environment_generated_at: string;
  priority_definition: string;
  records: Array<{
    district_id: string;
    district_name: string;
    horizon_month: number;
    target_start: string;
    target_end: string;
    research_indicator_probability: number;
    surveillance_priority_score: number;
    official_malaria_api: number;
    official_malaria_cases: number | null;
    official_burden_year: number;
    historical_district_month_context: number;
    forecast_precipitation_mean_mm: number;
    forecast_precipitation_p10_mm: number;
    forecast_precipitation_p90_mm: number;
    forecast_temperature_mean_c: number;
    forecast_temperature_p10_c: number;
    forecast_temperature_p90_c: number;
    environment_used_in_probability: boolean;
    source_url: string;
  }>;
}

export interface SyntheticForecastDistrict {
  synthetic_district_id: string;
  display_name: string;
  issue_date: string;
  target_date: string;
  probability: number;
  watermark: "SIMULATION_ONLY_NOT_ODISHA_RISK";
}

/** A single disease-group x horizon cell of the real occurrence model. */
export interface RealForecastCell {
  disease_group: string;
  horizon_weeks: number;
  reason_codes?: string[];
}

/** Fitted-artefact summary: which cells earned publication and which refused. */
export interface RealForecastSummary {
  model_version: string;
  generated_at: string;
  is_synthetic: false;
  is_incidence: false;
  is_case_count: false;
  quantity_statement: string;
  warning: string;
  published_cells: RealForecastCell[];
  refused_cells: RealForecastCell[];
}

/**
 * District probabilities for one cell. `status` is "published" only when the
 * model beat its seasonal climatology baseline; otherwise `districts` is empty
 * and `reason_codes` explains the refusal.
 */
export interface RealForecastMap {
  status: "published" | "insufficient_evidence";
  disease_group: string;
  horizon_weeks: number;
  issue_week?: string;
  reason_codes?: string[];
  districts: Array<{
    district_id: string;
    canonical_name: string;
    probability_reported_outbreak: number;
    seasonal_baseline_probability: number;
    observed_report_published: number;
  }>;
}

export interface SyntheticForecastReport {
  schema_version: string;
  watermark: "SIMULATION_ONLY_NOT_ODISHA_RISK";
  is_synthetic: true;
  seed: number;
  horizon_weeks: 1 | 2 | 4 | 8 | 12;
  target: string;
  models: string[];
  rolling_origins: Array<{
    origin_week_index: number;
    train_rows: number;
    test_rows: number;
    model_brier: number;
    seasonal_baseline_brier: number;
  }>;
  pooled: {
    model_brier: number;
    seasonal_baseline_brier: number;
    reliability: Array<{
      lower: number;
      upper: number;
      count: number;
      mean_probability: number;
      observed_fraction: number;
    }>;
    note: string;
  };
  latest_simulation_map: SyntheticForecastDistrict[];
  real_odisha_prediction_available: false;
}

export interface EvidenceLayerRecord {
  id?: string;
  district_id?: string | null;
  disease?: string | null;
  source_id?: string | null;
  source_snapshot_id?: string | null;
  created_at?: string | null;
  event?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface EvidenceLayerSnapshot {
  status: "loading" | "loaded" | "error";
  coverageState: string;
  records: EvidenceLayerRecord[];
  warnings: Warning[];
  deferrals: Array<Record<string, unknown>>;
}

export interface ReadinessCapability {
  capability: string;
  /**
   * `detail` is the deployment's own sentence about this capability. It is read
   * from either the state block or the capability row, and its absence is
   * rendered as absence — the interface never writes an explanation the register
   * did not supply.
   */
  state: { code: string; started_at?: string | null; detail?: string | null };
  operational: boolean;
  detail?: string | null;
}

export interface ReadinessData {
  profile: string;
  capabilities: ReadinessCapability[];
}

export interface SourceState {
  id: string;
  name: string;
  language: string;
  kind: string;
  url: string;
  state: "ready" | "registered_uncontacted" | "fallback" | "policy_pending" | "unavailable";
  note: string;
}

export interface DistrictValue {
  district: string;
  value: number | null;
  coverage: "observed" | "partial" | "unavailable";
}

export interface GeoFeatureCollection {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    properties: Record<string, unknown>;
    geometry: { type: "Polygon" | "MultiPolygon"; coordinates: unknown };
  }>;
}

export type AgentIntent =
  | "evidence_search"
  | "candidate_alerts"
  | "verified_events"
  | "forecast_request"
  | "incidence_request"
  | "data_audit"
  | "clinical_advice_request";

export type AgentAnswerState =
  | "records_returned"
  | "insufficient_evidence"
  | "risk_factor_outlook"
  | "forecast_probability"
  | "insufficient_training_data"
  | "not_observable_from_public_sources"
  | "ambiguous_scope"
  | "out_of_scope_clinical"
  | "audited_public_catalogue";

export interface AgentHistoryTurn {
  role: "user" | "assistant";
  content: string;
}

export interface AgentQueryRequest {
  question: string;
  district_id?: string;
  disease?: string;
  maximum_evidence?: number;
  /** Language the ANSWER should be written in. */
  target_language?: AnswerLanguage;
  /** Up to eight prior turns for resolving follow-ups such as “what about there?” */
  history?: AgentHistoryTurn[];
}

export interface AgentEvidenceCitation {
  signal_id?: string;
  source_id?: string;
  source_snapshot_id?: string;
  district_id?: string | null;
  disease?: string | null;
  assertion?: string;
  redacted_evidence?: string;
  retrieved_at?: string;
  /**
   * Digest of the stored content. On a deployment that does not retain source
   * content this is the same placeholder digest on every row, so it identifies
   * the retention policy rather than the record — see `evidenceDigest`.
   */
  content_sha256?: string;
  /** Digest of what this particular fetch returned; distinct per snapshot. */
  snapshot_content_sha256?: string | null;
  canonical_url?: string | null;
  access_path?: string | null;
  review_state?: string | null;
  review_decision?: string | null;
  is_fixture?: boolean;
}

/**
 * How the answer text was produced. `retrieval_template` is the deterministic
 * sentence the platform has always emitted; `generated` means a language model
 * wrote it from the retrieved evidence. The interface never guesses — an absent
 * field renders as "not reported by the API" rather than as a model claim.
 */
export type AgentGenerationMode = "generated" | "retrieval_template" | "policy_response" | "refusal";

export interface AgentTranslationTrace {
  status?: string;
  model?: string | null;
  pipeline?: string[];
  /** Answer text before translation, when the API translated it. */
  source_text?: string | null;
}

/** One scored candidate from the retriever, before the answer was written. */
export interface AgentRetrievedRecord {
  record_id?: string;
  score?: number;
  rank?: number;
}

/**
 * What the retriever did. `considered` is the size of the candidate pool, which
 * is larger than the evidence attached to the answer whenever the request caps
 * `maximum_evidence`.
 */
export interface AgentRetrievalTrace {
  state?: string | null;
  model?: string | null;
  query_language?: string | null;
  considered?: number | null;
  ranked?: AgentRetrievedRecord[];
}

/**
 * What the generator did. `cited_signal_ids` is the subset of the retrieved
 * evidence the answer text actually references; it is normally much smaller than
 * `considered_signal_ids`, and conflating the two overstates the grounding.
 */
export interface AgentGenerationTrace {
  answer?: string | null;
  answer_english?: string | null;
  answer_language?: string | null;
  generation_state?: string | null;
  cited_signal_ids?: string[];
  considered_signal_ids?: string[];
  model?: string | null;
  latency_ms?: number | null;
  prompt_evidence_count?: number | null;
  numeric_verification?: string | null;
  unverified_numbers?: unknown[];
  translation_state?: string | null;
  reason_code?: string | null;
  from_cache?: boolean | null;
}

export interface AgentQueryResult {
  intent: AgentIntent;
  answer_state: AgentAnswerState;
  answer: string;
  scope: {
    district_id: string | null;
    disease: string | null;
    question_language: string;
    conversation_context_used?: boolean;
  };
  /** Every record retrieved for the answer — not the records it cited. */
  evidence: AgentEvidenceCitation[];
  reason_codes: string[];

  /* --- fields the generating agent adds; all optional, all degrade cleanly --- */

  /**
   * Signal ids the answer text actually cites. An absent field means the API did
   * not report citations, which the interface states rather than papering over
   * by reusing the retrieval count.
   */
  citations?: string[] | null;
  retrieval?: AgentRetrievalTrace | null;
  generation?: AgentGenerationTrace | null;

  /** Identifier of the model that wrote `answer`, e.g. "Qwen/Qwen2.5-1.5B-Instruct". */
  model?: string | null;
  generation_mode?: AgentGenerationMode | string | null;
  /** Language `answer` is written in. */
  answer_language?: string | null;
  /** The pre-translation answer, when `answer` was translated. */
  answer_original?: string | null;
  translation?: AgentTranslationTrace | null;
  latency_ms?: number | null;
}

/* ------------------------------------------------------------------ translation */

export interface TranslateRequest {
  text: string;
  /** Omitted asks the service to detect from the source script. */
  source_language?: AnswerLanguage;
  target_language: AnswerLanguage;
}

export interface TranslationResult {
  status: "translated" | "passthrough" | "unavailable";
  /** Resolved source language; "und" when detection failed. */
  source_language: string;
  source_language_detected: boolean;
  target_language: AnswerLanguage;
  source_text: string;
  translated_text: string | null;
  /** Model identifier, e.g. "gaganmadan/IndicTrans2-preprint-ct2_int8". */
  model: string | null;
  /** Ordered stage labels, e.g. ["detect:en", "indictrans2-ct2-int8:eng_Latn>hin_Deva", "transliterate:hi>or"]. */
  pipeline: string[];
  latency_ms: number | null;
  /** Set when `status` is "unavailable", e.g. "translation_unavailable_source_language_only". */
  capability_code: string | null;
  is_synthetic: boolean;
}

/* ---------------------------------------------- current environmental conditions */

/**
 * One district on the current-conditions layer. The favourability field name is
 * read defensively because this payload is produced by the forecasting lane; a
 * row with no numeric field is treated as unmeasured, never as zero.
 */
export interface CurrentConditionsDistrict {
  district_id: string;
  canonical_name?: string | null;
  [key: string]: unknown;
}

export interface CurrentConditions {
  status?: "published" | "insufficient_evidence" | string;
  model_version?: string | null;
  layer_version?: string | null;
  generated_at?: string | null;
  as_of?: string | null;
  is_synthetic?: boolean;
  /** Either a prose statement or the artefact's `{kind, statement, ...}` block. */
  quantity?: string | Record<string, unknown> | null;
  quantity_statement?: string | null;
  warning?: string | null;
  warnings?: unknown[];
  capability_code?: string | null;
  reason_codes?: string[];
  asked_for?: string | null;
  target_series_supported_to?: string | null;
  message?: string | null;
  unlocked_by?: string | null;
  /** Metric name for the legend when the layer publishes. */
  metric?: string | null;
  /** Vintage block: how fresh the climate and IMD inputs are. */
  data_edge?: Record<string, unknown> | null;
  coverage?: Record<string, unknown> | null;
  districts: CurrentConditionsDistrict[];
}

/** Readiness of the authorised aggregate disease-data path for a real forecast. */
export interface OperationalForecastReadiness {
  status: string;
  eligible_for_training: boolean;
  reason_codes: string[];
  rows_all_vintages?: number;
  rows_latest_vintage?: number;
  diseases?: Record<string, {
    districts_observed: number;
    districts_meeting_history_and_completeness_floor: number;
    eligible: boolean;
  }>;
  template_path?: string;
  privacy_boundary?: string;
}

export interface OperationalForecastCell {
  disease: string;
  horizon_weeks: number;
  status: "qualified" | "insufficient_evidence" | string;
  reason_codes?: string[];
  examples: number;
}

export interface OperationalForecastSummary {
  status: string;
  model_version?: string;
  target_statement?: string;
  target_is_threshold_exceedance?: boolean;
  environmental_feature_state?: string;
  cells: OperationalForecastCell[];
}

export interface OperationalForecastMap {
  status: "published" | "insufficient_evidence" | string;
  disease: string;
  horizon_weeks: number;
  issue_week?: string;
  districts: Array<{
    district_id: string;
    disease: string;
    issue_week: string;
    target_week: string;
    horizon_weeks: number;
    probability_threshold_exceedance: number;
    outbreak_threshold_per_100k: number;
    threshold_version: string;
    latest_case_volume_completeness: number;
    latest_reporting_unit_completeness: number;
  }>;
  target_statement?: string;
}
