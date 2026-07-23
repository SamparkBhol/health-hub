import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, AlertTriangle, ArrowRight, ArrowUpRight, Bot, ExternalLink,
  Fingerprint, Grid3x3, Languages, Layers, Map as MapIcon, Menu, Radio, Search, ShieldCheck, Table2, X,
} from "lucide-react";
import { api } from "./api";
import { DistrictChoropleth, DistrictTable, MapLegend, buildScale } from "./DistrictMap";
import type { DistrictDatum } from "./DistrictMap";
import { AgentChat } from "./AgentChat";
import { TranslationPage } from "./Translate";
import { CurrentConditionsLayer } from "./RiskLayer";
import { OfficialHmisMap, OfficialMalariaMap, PublicResearchOutlook } from "./PublicHealth";
import { DiseaseChips, DiseaseMatrix, DistrictDetail, RetrievalTimeline, buildTimeline, districtIdFor } from "./Pattern";
import { isSynthetic, readDeferrals, typedState } from "./epistemics";
import type { Tone } from "./epistemics";
import { Chip, Notice, SyntheticBadge, TypedState, WarningStrip } from "./ui";
import { LAYERS, LAYER_ORDER } from "./layers";
import type { LayerSpec } from "./layers";
import { districtNames, fallbackSignals, fallbackSources } from "./fallback";
import { humanizeDisease } from "./format";
import type {
  EvidenceLayerRecord,
  EvidenceLayerSnapshot,
  GeoFeatureCollection,
  LayerType,
  OperationalForecastMap,
  OperationalForecastSummary,
  PublishedSignalMap,
  ReadinessCapability,
  Signal,
  SignalAssertion,
  SignalLanguage,
  SourceState,
  Warning,
} from "./types";

type Page =
  | "overview" | "assistant" | "translate" | "evidence"
  | "review" | "provenance" | "sources" | "forecast";

const nav: Array<{ id: Page; label: string }> = [
  { id: "overview", label: "Home" },
  { id: "assistant", label: "Agent" },
  { id: "evidence", label: "Maps" },
  { id: "translate", label: "Translate" },
  { id: "review", label: "Candidates" },
  { id: "provenance", label: "Provenance" },
  { id: "sources", label: "Sources" },
  { id: "forecast", label: "Forecast" },
];

function pageFromHash(hash: string): Page {
  const candidate = hash.replace(/^#/, "");
  return nav.some((item) => item.id === candidate) ? (candidate as Page) : "overview";
}

function formatTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "short", timeZone: "Asia/Kolkata" }).format(date);
}

function sourceHost(value: string): string {
  try {
    return new URL(value).hostname.replace(/^www\./, "");
  } catch {
    return value;
  }
}

/* ------------------------------------------------------------------- overview */

function CapabilityRegister({ capabilities, status }: { capabilities: ReadinessCapability[]; status: "loading" | "loaded" | "error" }) {
  return (
    <div className="register">
      <div className="register__head">
        <span className="eyebrow">Live capability register</span>
        <span className="register__source">GET /api/v1/readiness</span>
      </div>
      {status === "loading" && (
        <p className="register__pending">Reading the capability register from the API…</p>
      )}
      {status === "error" && (
        <TypedState code="unknown" capability="readiness" compact />
      )}
      {status === "loaded" && !capabilities.length && (
        <TypedState code="insufficient_evidence" capability="readiness" compact />
      )}
      {status === "loaded" && capabilities.length > 0 && (
        <ul className="register__list">
          {capabilities.map((entry) => {
            const copy = typedState(entry.state?.code);
            return (
              <li key={entry.capability} className={entry.operational ? "is-on" : "is-off"}>
                <span className="register__flag" aria-hidden="true">{entry.operational ? "ON" : "OFF"}</span>
                <span className="register__name">{entry.capability.replaceAll("_", " ")}</span>
                <Chip tone={entry.operational ? "ok" : copy.tone} size="sm">{entry.state?.code ?? "unknown"}</Chip>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function AppLauncher({ onNavigate }: { onNavigate: (page: Page) => void }) {
  const tools: Array<{
    page: Page;
    name: string;
    eyebrow: string;
    description: string;
    icon: React.ReactNode;
  }> = [
    {
      page: "assistant",
      name: "Ask the evidence agent",
      eyebrow: "Grounded answers",
      description: "Query Odisha public-health evidence in Odia, Hindi or English and inspect the cited source records.",
      icon: <Bot size={27} strokeWidth={2.75} aria-hidden="true" />,
    },
    {
      page: "evidence",
      name: "Explore Odisha maps",
      eyebrow: "30 districts",
      description: "Compare published disease signals, reviewed events, official catalogues and observed environmental conditions.",
      icon: <MapIcon size={27} strokeWidth={2.75} aria-hidden="true" />,
    },
    {
      page: "translate",
      name: "Translate evidence",
      eyebrow: "Three languages",
      description: "Read source spans across Odia, Hindi and English while keeping the original evidence beside the translation.",
      icon: <Languages size={27} strokeWidth={2.75} aria-hidden="true" />,
    },
  ];
  return (
    <section className="app-launcher" aria-labelledby="tools-heading">
      <div className="app-launcher__head">
        <span className="eyebrow">Operational workspaces</span>
        <h2 id="tools-heading">Choose a tool</h2>
      </div>
      <div className="app-launcher__grid">
        {tools.map((tool) => (
          <button key={tool.page} type="button" className="app-card" onClick={() => onNavigate(tool.page)}>
            <span className="app-card__icon">{tool.icon}</span>
            <span className="app-card__copy">
              <small>{tool.eyebrow}</small>
              <strong>{tool.name}</strong>
              <span>{tool.description}</span>
            </span>
            <ArrowRight size={21} strokeWidth={3} aria-hidden="true" />
          </button>
        ))}
      </div>
    </section>
  );
}

function Overview({
  onNavigate, apiMode, capabilities, capabilityStatus,
}: {
  onNavigate: (page: Page) => void;
  apiMode: "live" | "fixture" | "loading";
  capabilities: ReadinessCapability[];
  capabilityStatus: "loading" | "loaded" | "error";
}) {
  return (
    <>
      <section className="hero">
        <div className="hero__thesis">
          <div className="eyebrow eyebrow--flag">
            <Radio size={14} strokeWidth={3} aria-hidden="true" /> Odisha · Odia / Hindi / English public sources
          </div>
          <h1 className="display">
            Ask.<br />
            Translate.<br />
            <span className="display__hit">Map.</span><br />
            <span className="display__quiet">Verify the evidence.</span>
          </h1>
          <p className="hero__copy">
            Collect public-health information from registered Odia, Hindi and English sources, ask source-grounded
            questions, translate retained evidence and map published disease signals across Odisha&rsquo;s 30 districts.
            Case incidence and retrospective reporting probabilities remain visibly separate.
          </p>
          <div className="hero__actions">
            <button className="btn btn--primary" onClick={() => onNavigate("assistant")}>
              Ask the evidence agent <ArrowRight size={18} strokeWidth={3} aria-hidden="true" />
            </button>
            <button className="btn btn--ghost" onClick={() => onNavigate("evidence")}>
              Explore Odisha maps <ArrowUpRight size={18} strokeWidth={3} aria-hidden="true" />
            </button>
          </div>
          <p className="hero__runtime" aria-live="polite">
            <span className={`dot dot--${apiMode}`} aria-hidden="true" />
            {apiMode === "live"
              ? "Evidence API connected"
              : apiMode === "loading"
                ? "Checking the evidence API…"
                : "Fixture mode — no live claims"}
          </p>
        </div>
        <CapabilityRegister capabilities={capabilities} status={capabilityStatus} />
      </section>

      <AppLauncher onNavigate={onNavigate} />

      <section className="band" aria-label="Scope summary">
        <div><strong>30</strong><span>districts represented, with missing evidence kept visible</span></div>
        <div><strong>3</strong><span>languages: Odia, Hindi and English</span></div>
        <div><strong>5</strong><span>evidence layers kept separate by claim</span></div>
        <div><strong>1</strong><span>traceable workspace from source to map</span></div>
      </section>

      <section className="page-section" aria-labelledby="findings-heading">
        <div className="section-head">
          <div>
            <span className="eyebrow">Verified before build</span>
            <h2 id="findings-heading">Two findings that shaped the product</h2>
          </div>
          <Chip tone="mute">Audited 21 July 2026</Chip>
        </div>
        <div className="grid grid--2">
          <article className="slab finding finding--stop">
            <span className="finding__tag">Finding 01</span>
            <h3>The public history cannot train an Odisha outbreak model</h3>
            <p className="finding__figure">358 <small>Odisha rows across 14 years</small></p>
            <p>
              Only 2 of them are dengue. Across the national file, 28.0% of rows differ by more than one ISO week from
              the date-derived index even after year-boundary correction. EpiClim is a positive-only publication
              catalogue, not a district-week surveillance panel, so it has no negative weeks to learn from.
            </p>
            <p className="finding__proof"><Fingerprint size={15} strokeWidth={2.5} aria-hidden="true" /> Verified against MD5 <code>a6c961…f9f16</code></p>
          </article>
          <article className="slab finding finding--warn">
            <span className="finding__tag">Finding 02</span>
            <h3>Native full-body Odia recall is still unmeasured</h3>
            <p className="finding__figure">0 <small>native Odia test sets reported</small></p>
            <p>
              The published Oriya relevance evaluation we build on used IndicTrans2-translated English data, and
              downstream extraction ran on English translations of title and description. Preserving evidence in its
              original language is therefore not a solved problem here.
            </p>
            <p className="finding__proof"><ShieldCheck size={15} strokeWidth={2.5} aria-hidden="true" /> Published-method correction, not a performance claim</p>
          </article>
        </div>
      </section>

      <section className="page-section">
        <div className="section-head">
          <div>
            <span className="eyebrow">System contract</span>
            <h2>Autonomous collection, accountable decisions</h2>
          </div>
        </div>
        <ol className="pipeline">
          {[
            ["Acquire", "Registered URLs only, with a stored receipt per fetch"],
            ["Understand", "OCR, language routing, assertion state"],
            ["Protect", "Redact before anything is persisted"],
            ["Verify", "A person promotes an event, or nobody does"],
            ["Separate", "Five layers, published in five separate responses"],
          ].map(([title, text], index) => (
            <li key={title} className="pipeline__step">
              <span className="pipeline__index">{String(index + 1).padStart(2, "0")}</span>
              <strong>{title}</strong>
              <small>{text}</small>
            </li>
          ))}
        </ol>
      </section>
    </>
  );
}

/* ------------------------------------------------------------- evidence page */

interface CanonicalDistrict {
  id: string;
  name: string;
}

interface DisplayLayerRecord {
  id: string;
  title: string;
  district: string | null;
  districtId: string | null;
  disease: string | null;
  detail: string;
  source: string | null;
  sourceUrl: string | null;
  timestamp: string | null;
  metricValue: number | null;
  synthetic: boolean;
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function recordEvent(record: EvidenceLayerRecord): Record<string, unknown> {
  return record.event && typeof record.event === "object" ? record.event : {};
}

function canonicalDistricts(boundary: GeoFeatureCollection | null): CanonicalDistrict[] {
  if (!boundary) return districtNames.map((name) => ({ id: `OD-DIST-${name.toLowerCase()}`, name }));
  const resolved = boundary.features.flatMap((feature) => {
    const id = nonEmptyString(feature.properties.district_id);
    const name = nonEmptyString(feature.properties.canonical_name);
    return id && name ? [{ id, name }] : [];
  });
  return resolved.length ? resolved : districtNames.map((name) => ({ id: `OD-DIST-${name.toLowerCase()}`, name }));
}

function numericField(record: EvidenceLayerRecord, event: Record<string, unknown>): number | null {
  for (const key of ["map_value", "rate_per_100k", "reported_count", "case_count", "cases", "value"]) {
    const value = record[key] ?? event[key];
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) return value;
  }
  return null;
}

function displayLayerRecord(
  layer: LayerType, record: EvidenceLayerRecord, index: number, districts: CanonicalDistrict[],
): DisplayLayerRecord {
  const event = recordEvent(record);
  const districtId = nonEmptyString(record.district_id) ?? nonEmptyString(event.district_id);
  const canonical = districts.find((district) => district.id.toLowerCase() === districtId?.toLowerCase());
  const district = canonical?.name ?? nonEmptyString(record.district) ?? nonEmptyString(event.district) ?? districtId;
  const disease = humanizeDisease(nonEmptyString(record.disease) ?? nonEmptyString(event.disease));
  const source = nonEmptyString(record.source_id) ?? nonEmptyString(event.source_id);
  const sourceUrl = nonEmptyString(record.canonical_url)
    ?? nonEmptyString(record.registered_source_url)
    ?? nonEmptyString(event.canonical_url)
    ?? nonEmptyString(event.registered_source_url);
  const timestamp = nonEmptyString(record.created_at) ?? nonEmptyString(event.observed_at);
  const id = nonEmptyString(record.id) ?? `${layer}-${index}`;
  const outbreakId = nonEmptyString(event.outbreak_id);
  const week = typeof event.week === "number" ? event.week : null;
  const year = typeof event.year === "number" ? event.year : null;

  const detail = layer === "verified_event"
    ? "Promoted by a reviewer from candidate evidence."
    : layer === "official_event_catalogue"
      ? `${event.positive_only_catalogue === true ? "Positive-only catalogue entry; no row is not a zero." : "Public catalogue entry."}${outbreakId ? ` Publisher id ${outbreakId}.` : ""}${week && year ? ` Reporting week ${week}/${year}.` : ""}`
      : `Official weekly observation: ${typeof record.cases === "number" ? `${record.cases} reported cases; ` : ""}${typeof record.case_volume_completeness === "number" ? `${Math.round(record.case_volume_completeness * 100)}% case-volume completeness. ` : ""}${nonEmptyString(record.threshold_version) ? `Threshold ${nonEmptyString(record.threshold_version)}.` : ""}`;

  return {
    id,
    title: `${disease ?? "Disease not supplied"} · ${district ?? "District not supplied"}`,
    district,
    districtId: canonical?.id ?? null,
    disease,
    detail,
    source,
    sourceUrl,
    timestamp,
    metricValue: layer === "observed_surveillance" ? numericField(record, event) : null,
    synthetic: isSynthetic(record),
  };
}

/**
 * Converts a selected accumulation day into the `retrieved_to` bound the API
 * takes, so the choropleth redraws as the evidence base stood at the end of that
 * UTC day. `null` means the whole retained window.
 */
function retrievalBounds(cutoffDay: string | null): { retrieved_to?: string } {
  return cutoffDay ? { retrieved_to: `${cutoffDay}T23:59:59.999Z` } : {};
}

function LayerTabs({ value, onChange }: { value: LayerType; onChange: (next: LayerType) => void }) {
  const listRef = useRef<HTMLDivElement>(null);

  /** Roving tabindex: selection and focus always move together, or the tab order breaks. */
  const focusIndex = (index: number) => {
    const next = LAYER_ORDER[(index + LAYER_ORDER.length) % LAYER_ORDER.length];
    onChange(next);
    window.requestAnimationFrame(() => {
      listRef.current?.querySelector<HTMLButtonElement>(`[data-layer="${next}"]`)?.focus();
    });
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    const current = LAYER_ORDER.indexOf(value);
    const target = event.key === "ArrowRight" || event.key === "ArrowDown" ? current + 1
      : event.key === "ArrowLeft" || event.key === "ArrowUp" ? current - 1
        : event.key === "Home" ? 0
          : event.key === "End" ? LAYER_ORDER.length - 1
            : null;
    if (target === null) return;
    event.preventDefault();
    focusIndex(target);
  };

  return (
    <div className="tabs" role="tablist" aria-label="Evidence layers" ref={listRef} onKeyDown={onKeyDown}>
      {LAYER_ORDER.map((id, index) => (
        <button
          key={id}
          type="button"
          role="tab"
          id={`layer-tab-${id}`}
          data-layer={id}
          aria-selected={value === id}
          aria-controls="layer-panel"
          tabIndex={value === id ? 0 : -1}
          className={`tab${value === id ? " tab--on" : ""}`}
          onClick={() => onChange(id)}
        >
          <span className="tab__index">{String(index + 1).padStart(2, "0")}</span>
          <span className="tab__label">{LAYERS[id].tab}</span>
        </button>
      ))}
    </div>
  );
}

/** The signature block: every layer states its claim and its counter-claim at equal weight. */
function LayerContract({ spec }: { spec: LayerSpec }) {
  return (
    <div className="contract">
      <div className="contract__side contract__side--is">
        <span className="contract__tag">What this layer is</span>
        <p>{spec.is}</p>
      </div>
      <div className="contract__side contract__side--isnot">
        <span className="contract__tag">What this layer is <em>not</em></span>
        <p>{spec.isNot}</p>
      </div>
    </div>
  );
}

function CoverageBoard({ records }: { records: EvidenceLayerRecord[] }) {
  return (
    <div className="coverage-board">
      {records.map((record, index) => {
        const enabled = record.enabled === 1 || record.enabled === true;
        const lastSuccess = nonEmptyString(record.last_success_at);
        const id = nonEmptyString(record.source_id) ?? `source-${index}`;
        const sourceUrl = nonEmptyString(record.canonical_url);
        return (
          <article key={id} className={`slab coverage-card${enabled ? "" : " coverage-card--off"}`}>
            <div className="coverage-card__top">
              <Chip tone={enabled ? "ok" : "mute"} size="sm">{enabled ? "Route enabled" : "Route disabled"}</Chip>
              <span className="coverage-card__lang">{nonEmptyString(record.language) ?? "language not declared"}</span>
            </div>
            <h3>{nonEmptyString(record.name) ?? id}</h3>
            <p className="coverage-card__meta">
              {lastSuccess ? `Last successful receipt ${formatTime(lastSuccess)}` : "No successful collection receipt recorded"}
            </p>
            <p className="coverage-card__policy">{nonEmptyString(record.policy_state) ?? "Policy state not declared"}</p>
            <code>{id}</code>
            {sourceUrl && (
              <a className="source-link" href={sourceUrl} target="_blank" rel="noreferrer">
                {sourceHost(sourceUrl)} <ExternalLink size={13} strokeWidth={2.5} aria-hidden="true" />
              </a>
            )}
          </article>
        );
      })}
    </div>
  );
}

function SignalCard({ signal }: { signal: Signal }) {
  const tone: Tone = signal.assertion === "affirmed" ? "ok" : signal.assertion === "not_affirmed" ? "stop" : "warn";
  const disease = humanizeDisease(signal.disease) ?? "Unclassified health issue";
  return (
    <article className="record">
      <div className="record__top">
        <Chip tone={tone} size="sm">{signal.assertion.replaceAll("_", " ")}</Chip>
        <span className="record__lang">{signal.language.toUpperCase()}</span>
        {isSynthetic(signal) && <SyntheticBadge />}
      </div>
      <h3 lang={signal.language === "und" || signal.language === "mixed" ? undefined : signal.language}>
        {signal.titleOriginal ?? `${disease} evidence — ${signal.district}`}
      </h3>
      {signal.titleOriginal && <p className="record__translation">Interface label: {signal.title}</p>}
      <p className="record__meta">{signal.district} · {disease}</p>
      <blockquote className="record__quote">{signal.evidence}</blockquote>
      <p className="record__source">
        {signal.canonicalUrl ? (
          <a href={signal.canonicalUrl} target="_blank" rel="noreferrer">
            {signal.source} · {sourceHost(signal.canonicalUrl)} <ExternalLink size={12} strokeWidth={2.5} aria-hidden="true" />
          </a>
        ) : signal.source}
      </p>
    </article>
  );
}

function LayerRecordCard({ record, layer }: { record: DisplayLayerRecord; layer: LayerType }) {
  const label = layer === "verified_event" ? "verified event"
    : layer === "official_event_catalogue" ? "catalogue entry"
      : "observed aggregate";
  return (
    <article className="record">
      <div className="record__top">
        <Chip tone={layer === "verified_event" ? "ok" : "mute"} size="sm">{label}</Chip>
        {record.timestamp && <span className="record__lang">{formatTime(record.timestamp)}</span>}
        {record.synthetic && <SyntheticBadge />}
      </div>
      <h3>{record.title}</h3>
      <p className="record__detail">{record.detail}</p>
      {record.metricValue !== null && <p className="record__meta">Observed rate: {record.metricValue} per 100,000</p>}
      <p className="record__source">
        {record.sourceUrl ? (
          <a href={record.sourceUrl} target="_blank" rel="noreferrer">
            {record.source ? `${record.source} · ` : ""}{sourceHost(record.sourceUrl)} <ExternalLink size={12} strokeWidth={2.5} aria-hidden="true" />
          </a>
        ) : record.source ?? "Source not exposed by this response"}
      </p>
    </article>
  );
}

/**
 * Why every map on this page stops at district and never draws a taluka.
 *
 * The register carries a `tahasil_health_map` capability with its own state
 * code, and — where the deployment supplies one — a sentence explaining the
 * specific blocker. Both are rendered as received. If the register does not list
 * the capability at all, nothing is drawn here rather than an explanation this
 * client invented.
 */
function SubDistrictScope({
  capabilities, status,
}: {
  capabilities: ReadinessCapability[];
  status: "loading" | "loaded" | "error";
}) {
  if (status !== "loaded") return null;
  const entry = capabilities.find((item) => item.capability === "tahasil_health_map");
  if (!entry) return null;
  const detail = entry.state?.detail ?? entry.detail ?? null;
  return (
    <section className="scope-note" aria-labelledby="subdistrict-scope-heading">
      <div className="section-head section-head--tight">
        <div>
          <span className="eyebrow">Geographic resolution</span>
          <h2 id="subdistrict-scope-heading">Why these maps stop at district</h2>
        </div>
        <Chip tone={entry.operational ? "ok" : "warn"}>
          {entry.operational ? "tahasil layer available" : "district is the finest unit published"}
        </Chip>
      </div>
      <TypedState
        code={entry.state?.code ?? "unknown"}
        capability="tahasil_health_map"
        detail={detail}
      />
    </section>
  );
}

function Evidence({
  signals: initialSignals, boundary, apiMode, evidenceCoverage, layers, capabilities, capabilityStatus,
}: {
  signals: Signal[];
  boundary: GeoFeatureCollection | null;
  apiMode: "live" | "fixture" | "loading";
  evidenceCoverage: string;
  layers: Partial<Record<LayerType, EvidenceLayerSnapshot>>;
  capabilities: ReadinessCapability[];
  capabilityStatus: "loading" | "loaded" | "error";
}) {
  const [layer, setLayer] = useState<LayerType>("public_source_signal");
  const [view, setView] = useState<"map" | "table">("map");
  const [query, setQuery] = useState("");
  const [diseaseFilter, setDiseaseFilter] = useState("");
  const [languageFilter, setLanguageFilter] = useState<"" | SignalLanguage>("");
  const [assertionFilter, setAssertionFilter] = useState<SignalAssertion | "all">("affirmed");
  const [cutoffDay, setCutoffDay] = useState<string | null>(null);
  const [selectedDistrict, setSelectedDistrict] = useState<string | null>(null);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [publicSignals, setPublicSignals] = useState<Signal[]>(initialSignals);
  const [publicMap, setPublicMap] = useState<{
    data: PublishedSignalMap;
    coverageState: string;
    warnings: Warning[];
    deferrals: Array<Record<string, unknown>>;
  } | null>(null);
  const [publicMapStatus, setPublicMapStatus] = useState<"loading" | "loaded" | "error">("loading");

  const districts = useMemo(() => canonicalDistricts(boundary), [boundary]);
  const spec = LAYERS[layer];
  const snapshot = layers[layer] ?? { status: "error" as const, coverageState: "unknown", records: [], warnings: [], deferrals: [] };

  useEffect(() => {
    let active = true;
    setPublicMapStatus("loading");
    const filters = {
      disease: diseaseFilter || undefined,
      language: languageFilter || undefined,
      assertion: assertionFilter,
      ...retrievalBounds(cutoffDay),
    };
    Promise.allSettled([api.signals(filters), api.publishedSignalMap(filters)]).then(([signalResult, mapResult]) => {
      if (!active) return;
      if (signalResult.status === "fulfilled") {
        setPublicSignals(signalResult.value.data);
      } else if (apiMode === "fixture") {
        const disease = diseaseFilter.toLowerCase();
        setPublicSignals(initialSignals.filter((signal) => (
          (!disease || signal.disease.toLowerCase().replaceAll(" ", "_") === disease)
          && (!languageFilter || signal.language === languageFilter)
          && (assertionFilter === "all" || signal.assertion === assertionFilter)
        )));
      } else {
        setPublicSignals([]);
      }
      if (mapResult.status === "fulfilled") {
        setPublicMap({
          data: mapResult.value.data,
          coverageState: mapResult.value.context.coverage_state,
          warnings: mapResult.value.warnings ?? [],
          deferrals: mapResult.value.deferrals ?? [],
        });
        setPublicMapStatus("loaded");
      } else {
        setPublicMap(null);
        setPublicMapStatus("error");
      }
    });
    return () => { active = false; };
  }, [apiMode, assertionFilter, cutoffDay, diseaseFilter, initialSignals, languageFilter]);

  const displayRecords = useMemo(
    () => snapshot.records.map((record, index) => displayLayerRecord(layer, record, index, districts)),
    [districts, layer, snapshot.records],
  );

  const observedUsesValues = layer === "observed_surveillance" && displayRecords.some((record) => record.metricValue !== null);

  const districtData = useMemo<DistrictDatum[]>(() => {
    const counts = new Map<string, number>();
    if (layer === "public_source_signal") {
      for (const row of publicMap?.data.districts ?? []) counts.set(row.district_id, row.published_signal_count);
    } else if (spec.mappable) {
      for (const record of displayRecords) {
        if (!record.districtId) continue;
        if (observedUsesValues && record.metricValue === null) continue;
        counts.set(record.districtId, (counts.get(record.districtId) ?? 0) + (observedUsesValues ? record.metricValue ?? 0 : 1));
      }
    }
    return districts.map((district) => ({
      districtId: district.id,
      name: district.name,
      value: counts.has(district.id) ? counts.get(district.id) ?? null : null,
    }));
  }, [districts, displayRecords, layer, observedUsesValues, publicMap, spec.mappable]);

  const { bins, lowers } = useMemo(() => buildScale(districtData.map((entry) => entry.value)), [districtData]);
  const noDataCount = districtData.filter((entry) => entry.value === null).length;

  /** The accumulation axis is built from every retained receipt, not the filtered view. */
  const timeline = useMemo(() => buildTimeline(initialSignals), [initialSignals]);

  const districtSignals = useMemo(
    () => (selectedDistrict
      ? publicSignals.filter((signal) => districtIdFor(signal.district, districts) === selectedDistrict)
      : []),
    [districts, publicSignals, selectedDistrict],
  );
  const selectedAggregate = publicMap?.data.districts.find((row) => row.district_id === selectedDistrict) ?? null;
  const selectedName = districts.find((district) => district.id === selectedDistrict)?.name
    ?? selectedDistrict?.replace(/^OD-DIST-/i, "")
    ?? "";

  const meta = layer === "public_source_signal"
    ? {
      status: publicMapStatus,
      coverageState: publicMap?.coverageState ?? evidenceCoverage,
      warnings: publicMap?.warnings ?? [],
      deferrals: publicMap?.deferrals ?? [],
    }
    : {
      status: snapshot.status,
      coverageState: snapshot.coverageState,
      warnings: snapshot.warnings,
      deferrals: snapshot.deferrals,
    };

  const deferrals = useMemo(() => readDeferrals(meta.deferrals), [meta.deferrals]);
  const layerSynthetic = layer === "public_source_signal"
    ? publicMap?.data.fixture_mode === "fixture_only" || meta.coverageState === "fixture_fallback"
    : snapshot.records.length > 0 && snapshot.records.every(isSynthetic);

  const normalizedQuery = query.trim().toLowerCase();
  const filteredSignals = publicSignals.filter((signal) =>
    `${signal.title} ${signal.titleOriginal ?? ""} ${signal.district} ${signal.disease}`.toLowerCase().includes(normalizedQuery));
  const filteredRecords = displayRecords.filter((record) =>
    `${record.title} ${record.district ?? ""} ${record.disease ?? ""} ${record.source ?? ""}`.toLowerCase().includes(normalizedQuery));
  const sidebarCount = layer === "public_source_signal" ? filteredSignals.length : filteredRecords.length;
  const coverageCopy = typedState(meta.coverageState);
  const highlightedRow = districtData.find((entry) => entry.districtId === highlighted) ?? null;
  /** Drill-down only exists where the sidebar has per-district records to show. */
  const drillable = layer === "public_source_signal";

  return (
    <section className="page-section evidence">
      <div className="page-head">
        <div>
          <span className="eyebrow">Interactive district workspaces</span>
          <h1>Odisha health maps</h1>
          <p>
            Where health information gets published across Odisha&rsquo;s 30 districts, by disease tag and by retrieval
            day. Five layers, five separate API responses, never merged into one score — switch a layer and the map,
            the legend and the caption all change together.
          </p>
        </div>
        <div className="page-head__status">
          <Chip tone={coverageCopy.tone}>{meta.coverageState.replaceAll("_", " ")}</Chip>
          {layerSynthetic && <SyntheticBadge label="Synthetic layer" />}
        </div>
      </div>

      <OfficialMalariaMap boundary={boundary} districts={districts} />

      <OfficialHmisMap boundary={boundary} districts={districts} />

      <SubDistrictScope capabilities={capabilities} status={capabilityStatus} />

      <LayerTabs value={layer} onChange={(next) => { setLayer(next); setHighlighted(null); }} />

      <div id="layer-panel" role="tabpanel" aria-labelledby={`layer-tab-${layer}`}>
        <LayerContract spec={spec} />

        {layerSynthetic && (
          <Notice tone="sim" title="Every value in this layer is synthetic test data">
            <p>
              These rows come from bundled fixtures, not from a live collection receipt. They exercise the pipeline
              shape. They describe no real district, disease or event.
            </p>
          </Notice>
        )}

        <WarningStrip warnings={meta.warnings} />

        {deferrals.map((deferral) => (
          <TypedState
            key={`${deferral.capability}-${deferral.code}`}
            code={deferral.code}
            reasonCode={deferral.reasonCode}
            capability={deferral.capability}
            compact
          />
        ))}

        {layer === "public_source_signal" && (
          <>
            <div className="patternbar">
              <div className="patternbar__head">
                <span className="eyebrow">Disease pattern controls</span>
                <span className="patternbar__hint">Picking a tag redraws the choropleth from a fresh API query.</span>
              </div>
              <DiseaseChips
                signals={initialSignals}
                value={diseaseFilter}
                onChange={(next) => { setDiseaseFilter(next); setSelectedDistrict(null); }}
              />
              <RetrievalTimeline
                buckets={timeline}
                cutoff={cutoffDay}
                onCutoff={(day) => { setCutoffDay(day); setSelectedDistrict(null); }}
              />
            </div>
            <div className="filters" role="group" aria-label="Published signal filters">
            <label>
              <span>Source language</span>
              <select value={languageFilter} onChange={(event) => setLanguageFilter(event.target.value as "" | SignalLanguage)}>
                <option value="">All languages</option>
                <option value="or">Odia</option>
                <option value="hi">Hindi</option>
                <option value="en">English</option>
                <option value="mixed">Mixed</option>
                <option value="und">Undetermined</option>
              </select>
            </label>
            <label>
              <span>Assertion</span>
              <select value={assertionFilter} onChange={(event) => setAssertionFilter(event.target.value as SignalAssertion | "all")}>
                <option value="affirmed">Affirmed current mentions</option>
                <option value="all">All assertion states</option>
                <option value="not_affirmed">Not affirmed</option>
                <option value="speculative">Speculative</option>
                <option value="non_current">Non-current</option>
              </select>
            </label>
            <p className="filters__note">
              {cutoffDay
                ? `Map is drawn as of ${cutoffDay} (retrieval time). Later receipts are excluded.`
                : "Map covers every retained receipt. The time axis above uses system retrieval time — not article date, symptom onset or case date."}
            </p>
            </div>
          </>
        )}

        {!spec.mappable ? (
          <>
            <Notice tone="warn" title="This layer has no district geometry">
              <p>{spec.isNot}</p>
            </Notice>
            {snapshot.status === "loading" && <p className="pending">Loading the registered source set…</p>}
            {snapshot.status === "error" && <TypedState code="unknown" capability={layer} />}
            {snapshot.status === "loaded" && (snapshot.records.length
              ? <CoverageBoard records={snapshot.records} />
              : <TypedState code="insufficient_evidence" capability={layer} />)}
          </>
        ) : (
          <>
            <div className="tally">
              <p className="tally__figure">
                <strong>{noDataCount}</strong>
                <span>of {districtData.length} districts returned no record on this layer</span>
              </p>
              <div className="tally__controls">
                <div className="toggle" role="group" aria-label="Map or table view">
                  <button type="button" className={`toggle__btn${view === "map" ? " toggle__btn--on" : ""}`} aria-pressed={view === "map"} onClick={() => setView("map")}>
                    <MapIcon size={16} strokeWidth={2.75} aria-hidden="true" /> Map
                  </button>
                  <button type="button" className={`toggle__btn${view === "table" ? " toggle__btn--on" : ""}`} aria-pressed={view === "table"} onClick={() => setView("table")}>
                    <Table2 size={16} strokeWidth={2.75} aria-hidden="true" /> Table
                  </button>
                </div>
              </div>
            </div>

            {meta.status === "error" && (
              <Notice tone="stop" title="This layer's endpoint did not answer">
                <p>
                  No district value is drawn from cached or fixture data in its place. An unavailable response is not
                  an empty district.
                </p>
              </Notice>
            )}

            <div className="workspace">
              <div className="workspace__map">
                {layer === "public_source_signal" && view === "map" && (
                  <p className="stamp" aria-hidden="true">
                    <span>Published evidence</span>
                    <strong>NOT CASE COUNTS</strong>
                  </p>
                )}
                {meta.status === "loading" ? (
                  <div className="pending pending--block">
                    Requesting the {spec.name} aggregate. The map stays empty until a real response arrives.
                  </div>
                ) : (
                  <>
                    {view === "map" && (
                      <DistrictChoropleth
                        boundary={boundary}
                        data={districtData}
                        bins={bins}
                        lowers={lowers}
                        metric={spec.metric}
                        describedBy="district-value-table"
                        highlighted={highlighted}
                        onHighlight={setHighlighted}
                        onSelect={drillable ? setSelectedDistrict : undefined}
                        selected={selectedDistrict}
                      />
                    )}
                    <DistrictTable
                      id="district-value-table"
                      data={districtData}
                      bins={bins}
                      lowers={lowers}
                      metric={spec.metric}
                      unit={spec.unit}
                      hidden={view === "map"}
                      highlighted={highlighted}
                      onHighlight={setHighlighted}
                      onSelect={drillable ? setSelectedDistrict : undefined}
                      selected={selectedDistrict}
                    />
                    <MapLegend bins={bins} metric={spec.metric} noDataCount={noDataCount} totalCount={districtData.length} />
                    <p className="readout" aria-live="polite">
                      {highlightedRow
                        ? `${highlightedRow.name}: ${highlightedRow.value === null ? "no returned record — unmeasured, not zero" : `${highlightedRow.value} ${spec.unit}`}`
                        : drillable
                          ? "Point at a district to read its value; select one to open its evidence."
                          : "Point at a district, or switch to the table, to read a value."}
                    </p>
                  </>
                )}
              </div>

              <aside className="workspace__side" aria-label={`${spec.name} records`}>
                {drillable && selectedDistrict ? (
                  <DistrictDetail
                    districtId={selectedDistrict}
                    districtName={selectedName}
                    signals={districtSignals}
                    firstRetrievedAt={selectedAggregate?.first_retrieved_at}
                    lastRetrievedAt={selectedAggregate?.last_retrieved_at}
                    onClear={() => setSelectedDistrict(null)}
                  />
                ) : (
                  <>
                    <div className="side-search">
                      <label className="sr-only" htmlFor="record-search">Search returned records by district, disease or source</label>
                      <Search size={16} strokeWidth={2.75} aria-hidden="true" />
                      <input id="record-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search district, disease, source" />
                    </div>
                    <p className="side-count"><span>Returned records</span><strong>{sidebarCount}</strong></p>
                    <div className="side-list">
                      {layer === "public_source_signal"
                        ? filteredSignals.map((signal) => <SignalCard key={signal.id} signal={signal} />)
                        : filteredRecords.map((record) => <LayerRecordCard key={record.id} record={record} layer={layer} />)}
                      {!sidebarCount && meta.status !== "loading" && (
                        <div className="side-empty">
                          <strong>{spec.emptyTitle}</strong>
                          <p>{spec.emptyBody}</p>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </aside>
            </div>

            {layer === "public_source_signal" && (
              <section className="patternview" aria-labelledby="pattern-heading">
                <div className="section-head section-head--tight">
                  <div>
                    <span className="eyebrow">One map cannot carry two variables</span>
                    <h2 id="pattern-heading">District × disease pattern</h2>
                  </div>
                  <Chip tone="mute">{initialSignals.length} retained records</Chip>
                </div>
                <DiseaseMatrix signals={initialSignals} districts={districts} />
              </section>
            )}
          </>
        )}

        <Notice tone="warn" title="Ascertainment bias runs through every layer above">
          <p>
            {layer === "observed_surveillance"
              ? "This layer comes from authorised aggregate surveillance, not media. Its uncertainty is reporting completeness: a low completeness value changes what a low rate means, so it is displayed beside every observation and never converted to zero."
              : spec.mappable
              ? "Districts differ in how much gets published about them, in which language, and how quickly. A darker district on this map is better covered by the registered sources. Reading it as sicker is the specific mistake this interface is built to prevent."
              : "Every district stays unmeasured until a district-scoped observation receipt exists. A source that is enabled and healthy still tells you nothing about any particular district."}
          </p>
        </Notice>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------- other pages */

function Review({ signals }: { signals: Signal[] }) {
  const [selected, setSelected] = useState(0);
  const current = signals[selected] ?? signals[0];
  const currentDisease = humanizeDisease(current?.disease) ?? "Unclassified health issue";
  return (
    <section className="page-section">
      <div className="page-head">
        <div>
          <span className="eyebrow">Public read-only surface</span>
          <h1>Candidate evidence</h1>
          <p>
            A preview of public candidate records. This is not the authenticated review queue and it cannot promote
            an event.
          </p>
        </div>
        <div className="page-head__status">
          {signals.length > 0 && signals.every(isSynthetic) && <SyntheticBadge label="Synthetic queue" />}
          <Chip tone="mute">Read-only</Chip>
        </div>
      </div>
      {current ? (
        <div className="review">
          <div className="review__queue">
            <p className="side-count"><span>Candidates</span><strong>{signals.length}</strong></p>
            <ul>
              {signals.map((signal, index) => (
                <li key={signal.id}>
                  <button
                    type="button"
                    className={`queue-item${index === selected ? " queue-item--on" : ""}`}
                    aria-current={index === selected ? "true" : undefined}
                    onClick={() => setSelected(index)}
                  >
                    <span className={`assert assert--${signal.assertion}`} aria-hidden="true" />
                    <span>
                      <strong>{humanizeDisease(signal.disease) ?? "Unclassified"} · {signal.district}</strong>
                      <small>{signal.source}</small>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
          <article className="slab review__doc">
            <div className="review__toolbar">
              <div>
                <Chip tone="warn" size="sm">{current.language.toUpperCase()} source</Chip>
                <Chip size="sm">{current.assertion.replaceAll("_", " ")}</Chip>
                {isSynthetic(current) && <SyntheticBadge />}
              </div>
              <span>{formatTime(current.retrievedAt)}</span>
            </div>
            <h2 lang={current.language === "und" || current.language === "mixed" ? undefined : current.language}>
              {current.titleOriginal ?? `${currentDisease} evidence — ${current.district}`}
            </h2>
            <p className="review__source">
              {current.canonicalUrl ? (
                <a href={current.canonicalUrl} target="_blank" rel="noreferrer">
                  {current.source} · {sourceHost(current.canonicalUrl)} <ExternalLink size={13} strokeWidth={2.5} aria-hidden="true" />
                </a>
              ) : current.source} · {current.district}
            </p>
            <blockquote className="review__evidence">
              <span>Exact redacted evidence span</span>
              <p>{current.evidence}</p>
            </blockquote>
            <dl className="kv">
              <div><dt>Disease</dt><dd>{currentDisease}</dd></div>
              <div><dt>Place</dt><dd>{current.district}</dd></div>
              <div><dt>Assertion</dt><dd>{current.assertion.replaceAll("_", " ")}</dd></div>
              <div><dt>Digest</dt><dd><code>{current.hash ? `${current.hash.slice(0, 16)}…` : "not exposed"}</code></dd></div>
            </dl>
            <div className="review__decisions">
              <button type="button" disabled>Reject</button>
              <button type="button" disabled>Needs evidence</button>
              <button type="button" disabled className="is-accept">Verify event</button>
            </div>
            <p className="review__note">
              These controls are disabled on purpose. Promotion happens in the authenticated review service, which
              this public client never calls.
            </p>
          </article>
        </div>
      ) : (
        <TypedState code="insufficient_evidence" capability="candidate_signals" />
      )}
    </section>
  );
}

function Provenance({ signals }: { signals: Signal[] }) {
  const first = signals[0];
  const steps: Array<[string, string, string]> = [
    ["Registered source", first?.source ?? "No signal returned", first?.canonicalUrl ?? "Canonical URL not exposed"],
    ["Source snapshot", first?.sourceSnapshotId ?? "Snapshot id not exposed", first?.accessPath ?? "Access path not exposed"],
    ["Retrieval receipt", first?.hash ?? "SHA-256 not exposed", first ? `Retrieved ${formatTime(first.retrievedAt)}` : "Retrieval time unavailable"],
    ["Redacted evidence", first?.evidence ?? "No evidence returned", first ? `${humanizeDisease(first.disease) ?? "Unclassified"} · ${first.district} · ${first.assertion}` : "Extraction fields unavailable"],
    ["Public review state", first?.reviewState ?? "Review state unavailable", "A signal field, not the private decision log"],
  ];
  return (
    <section className="page-section page-section--narrow">
      <div className="page-head">
        <div>
          <span className="eyebrow">Public receipt fields</span>
          <h1>Provenance chain</h1>
          <p>Only the provenance the API actually returns. A field it does not expose stays marked unavailable.</p>
        </div>
      </div>
      <ol className="chain">
        {steps.map(([title, value, note], index) => (
          <li key={title} className="chain__step">
            <span className="chain__index">{String(index + 1).padStart(2, "0")}</span>
            <div>
              <span className="chain__title">{title}</span>
              <strong>{value}</strong>
              <small>{note}</small>
            </div>
          </li>
        ))}
      </ol>
      <div className="slab callout">
        <strong>Public-view boundary</strong>
        <p>
          This page shows redacted evidence text, source and snapshot identifiers where available, retrieval time,
          access path and digest. Parser versions, raw stored content and internal evidence offsets are not claimed
          here, because the public signal response does not return them.
        </p>
      </div>
    </section>
  );
}

function Sources({ sources }: { sources: SourceState[] }) {
  const tone: Record<SourceState["state"], Tone> = {
    ready: "ok", registered_uncontacted: "mute", fallback: "warn", policy_pending: "warn", unavailable: "stop",
  };
  return (
    <section className="page-section">
      <div className="page-head">
        <div>
          <span className="eyebrow">Declared acquisition surface</span>
          <h1>Source registry</h1>
          <p>
            Only registered URLs are fetched. A source being public does not by itself authorise archiving,
            redistribution or model training.
          </p>
        </div>
        <Chip tone="mute">{sources.length} registered routes</Chip>
      </div>

      <Notice tone="warn" title="The IDSP origin outage is handled, not hidden">
        <p>
          The connector tries the canonical live origin first, then a frozen Wayback <code>id_</code> capture. Week 9
          of 2026 is the newest report verified retrievable on 21 July 2026. Every response is snapshotted and hashed.
        </p>
      </Notice>

      <div className="data-table-wrap">
        <table className="data-table data-table--sources">
          <caption className="sr-only">Registered sources with language, collection state and acquisition note</caption>
          <thead>
            <tr><th scope="col">Source</th><th scope="col">Language / format</th><th scope="col">State</th><th scope="col">Acquisition note</th><th scope="col">Source URL</th></tr>
          </thead>
          <tbody>
            {sources.map((source) => (
              <tr key={source.id}>
                <th scope="row"><strong>{source.name}</strong><code>{source.id}</code></th>
                <td>{source.language}<small>{source.kind}</small></td>
                <td><Chip tone={tone[source.state]} size="sm">{source.state.replaceAll("_", " ")}</Chip></td>
                <td className="data-table__note">{source.note}</td>
                <td>
                  <a className="source-link" href={source.url} target="_blank" rel="noreferrer" aria-label={`Open ${source.name} in a new tab`}>
                    {sourceHost(source.url)} <ExternalLink size={14} strokeWidth={2.5} aria-hidden="true" />
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="section-head section-head--tight">
        <div>
          <span className="eyebrow">Measurement status</span>
          <h2>Components not yet benchmarked</h2>
        </div>
        <Chip tone="warn">No performance claim</Chip>
      </div>
      <div className="grid grid--5">
        {[
          ["OCR", "Odia and Hindi error on real notices is unmeasured", "Low-confidence pages go to language review."],
          ["Redaction", "PII recall is unvalidated", "Mitigation and isolation, not a guarantee."],
          ["Dedup", "Cross-language merging is unvalidated", "The map counts documents, not unique events."],
          ["Extraction", "Entity, assertion and place accuracy is unvalidated", "Ambiguity abstains and goes to review."],
          ["Signal value", "Lead-time advantage is unproven", "Media can repeat and lag an official notice."],
        ].map(([name, state, action]) => (
          <article key={name} className="slab limit">
            <span>{name}</span>
            <strong>{state}</strong>
            <p>{action}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

/**
 * The only operational forecast surface: qualified models trained from the
 * authorised, versioned district-week aggregate and current environmental
 * feature block. The public positive-only catalogue never reaches this UI.
 */
function OperationalForecastModel() {
  const [summary, setSummary] = useState<OperationalForecastSummary | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [map, setMap] = useState<OperationalForecastMap | null>(null);
  const [state, setState] = useState<"loading" | "loaded" | "error">("loading");
  const [mapState, setMapState] = useState<"idle" | "loading" | "loaded" | "error">("idle");

  useEffect(() => {
    let active = true;
    api.operationalForecastSummary()
      .then((response) => {
        if (!active) return;
        setSummary(response.data);
        const first = response.data.cells.find((cell) => cell.status === "qualified");
        setSelected(first ? `${first.disease}:${first.horizon_weeks}` : null);
        setState("loaded");
      })
      .catch(() => { if (active) setState("error"); });
    return () => { active = false; };
  }, []);

  const selectedCell = useMemo(
    () => (summary?.cells ?? []).find((cell) => `${cell.disease}:${cell.horizon_weeks}` === selected) ?? null,
    [selected, summary],
  );

  useEffect(() => {
    if (!selectedCell) { setMap(null); setMapState("idle"); return; }
    let active = true;
    setMap(null);
    setMapState("loading");
    api.operationalForecastMap(selectedCell.disease, selectedCell.horizon_weeks)
      .then((response) => { if (active) { setMap(response.data); setMapState("loaded"); } })
      .catch(() => { if (active) setMapState("error"); });
    return () => { active = false; };
  }, [selectedCell]);

  const ranked = useMemo(
    () => [...(map?.districts ?? [])].sort(
      (left, right) => right.probability_threshold_exceedance - left.probability_threshold_exceedance,
    ),
    [map],
  );
  const qualified = (summary?.cells ?? []).filter((cell) => cell.status === "qualified");

  return (
    <div className="realmodel">
      <div className="realmodel__head">
        <div>
          <span className="eyebrow">Authorised surveillance + environment</span>
          <h2>Disease threshold forecast</h2>
        </div>
        <Chip tone={qualified.length ? "ok" : "warn"}>{qualified.length ? "qualified model" : "data/model gate"}</Chip>
      </div>

      {state === "loading" && <p className="pending pending--block">Checking the authorised model register…</p>}
      {state === "error" && <TypedState code="source_temporarily_unavailable" capability="authorised_district_week_surveillance_forecast" compact />}
      {state === "loaded" && !qualified.length && (
        <Notice tone="warn" title="No disease probability is issued yet">
          <p>
            The agent is ready to train from the authorised no-PII district-week export, but no disease/horizon has yet cleared
            the history, completeness, rolling-origin and calibration gates. The environmental map remains available as context.
          </p>
          {(summary?.cells ?? []).length > 0 && (
            <ul className="refusal__codes">
              {summary?.cells.map((cell) => <li key={`${cell.disease}-${cell.horizon_weeks}`}><code>{cell.disease} · {cell.horizon_weeks}w · {cell.status}</code></li>)}
            </ul>
          )}
        </Notice>
      )}

      {qualified.length > 0 && (
        <>
          <p className="realmodel__quantity">
            {summary?.target_statement ?? "Probability of crossing the registered disease-specific threshold."}
          </p>
          <label className="filters__label">
            <span>Qualified disease and horizon</span>
            <select value={selected ?? ""} onChange={(event) => setSelected(event.target.value)}>
              {qualified.map((cell) => <option key={`${cell.disease}:${cell.horizon_weeks}`} value={`${cell.disease}:${cell.horizon_weeks}`}>{humanizeDisease(cell.disease) ?? cell.disease} · {cell.horizon_weeks} weeks</option>)}
            </select>
          </label>
          {mapState === "loading" && <p className="pending pending--block">Scoring the current district feature rows…</p>}
          {mapState === "error" && <TypedState code="insufficient_evidence" capability="current_authorised_forecast" compact />}
          {map?.status === "published" && (
            <>
              <p className="realmodel__issue">Issued for week of <strong>{map.issue_week}</strong> · target {map.horizon_weeks} weeks ahead · {ranked.length} complete district rows</p>
              <ul className="realmodel__bars">
                {ranked.slice(0, 10).map((district) => (
                  <li key={district.district_id}>
                    <span className="realmodel__name">{district.district_id.replace(/^OD-DIST-/, "")}</span>
                    <span className="realmodel__track"><span className="realmodel__fill" style={{ width: `${Math.max(1.5, district.probability_threshold_exceedance * 100)}%` }} /></span>
                    <span className="realmodel__value">{(district.probability_threshold_exceedance * 100).toFixed(1)}%<small> threshold {district.outbreak_threshold_per_100k}/100k · {Math.round(district.latest_case_volume_completeness * 100)}% complete</small></span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </>
      )}
    </div>
  );
}

function Forecast({ boundary }: { boundary: GeoFeatureCollection | null }) {
  const districts = useMemo(() => canonicalDistricts(boundary), [boundary]);

  return (
    <section className="page-section page-section--narrow">
      <div className="page-head">
        <div>
          <span className="eyebrow">Official burden + seasonal ensemble + authorised path</span>
          <h1>Odisha outbreak outlook</h1>
          <p>
            Explore a real 1–3 month public-data malaria outlook, its rolling-origin model comparison,
            the live district climate context and—when sponsor data clear the gate—the separate
            operational threshold forecast.
          </p>
        </div>
        <Chip tone="ok">Real public data</Chip>
      </div>

      <Notice tone="ok" title="Three outputs, three meanings">
        <p>
          The public research outlook ranks surveillance attention, the environmental map describes
          current risk factors, and the authorised model—when qualified—predicts a registered disease
          threshold. The interface never renames one as another.
        </p>
      </Notice>

      <PublicResearchOutlook boundary={boundary} districts={districts} />

      <CurrentConditionsLayer boundary={boundary} districts={districts} />

      <OperationalForecastModel />

      <div className="section-head section-head--tight"><div><h2>Operational promotion gate</h2></div></div>
      <div className="grid grid--2">
        {[
          ["Enough independent seasons", "Revision-aware routine surveillance with reporting completeness and stable case definitions; public HMIS records do not replace this."],
          ["Future climate treated as uncertain", "At 12 to 13 weeks, rainfall and temperature must themselves be forecast. Using observed future climate leaks information."],
          ["Disease-specific horizons", "Score 1, 2, 4, 8 and 12 weeks separately. One horizon cannot serve dengue, malaria and enteric disease at once."],
          ["Wins against real baselines", "Beat weekly climatology, persistence and a seasonal GLM on proper scores, with reliability diagnostics."],
        ].map(([title, body]) => (
          <article key={title} className="slab requirement"><strong>{title}</strong><p>{body}</p></article>
        ))}
      </div>
    </section>
  );
}

/* --------------------------------------------------------------------- shell */

const loadableLayers: LayerType[] = ["verified_event", "official_event_catalogue", "observed_surveillance", "coverage"];

function loadingLayers(): Partial<Record<LayerType, EvidenceLayerSnapshot>> {
  return Object.fromEntries(loadableLayers.map((layer) => [layer, {
    status: "loading", coverageState: "unknown", records: [], warnings: [], deferrals: [],
  }])) as Partial<Record<LayerType, EvidenceLayerSnapshot>>;
}

function layerSnapshotFrom(result: PromiseSettledResult<Awaited<ReturnType<typeof api.layer>>>): EvidenceLayerSnapshot {
  if (result.status === "rejected") {
    return { status: "error", coverageState: "unknown", records: [], warnings: [], deferrals: [] };
  }
  return {
    status: "loaded",
    coverageState: result.value.context.coverage_state,
    records: Array.isArray(result.value.data) ? result.value.data : [],
    warnings: result.value.warnings ?? [],
    deferrals: result.value.deferrals ?? [],
  };
}

export default function App() {
  const [page, setPage] = useState<Page>(() => pageFromHash(window.location.hash));
  const [menuOpen, setMenuOpen] = useState(false);
  const [apiMode, setApiMode] = useState<"live" | "fixture" | "loading">("loading");
  const [signals, setSignals] = useState<Signal[]>(fallbackSignals);
  const [sources, setSources] = useState<SourceState[]>(fallbackSources);
  const [boundary, setBoundary] = useState<GeoFeatureCollection | null>(null);
  const [evidenceCoverage, setEvidenceCoverage] = useState("unknown");
  const [capabilities, setCapabilities] = useState<ReadinessCapability[]>([]);
  const [capabilityStatus, setCapabilityStatus] = useState<"loading" | "loaded" | "error">("loading");
  const [layers, setLayers] = useState<Partial<Record<LayerType, EvidenceLayerSnapshot>>>(loadingLayers);

  useEffect(() => {
    const sync = () => {
      if (window.location.hash === "#main") {
        document.getElementById("main")?.focus();
        return;
      }
      const next = pageFromHash(window.location.hash);
      if (window.location.hash && window.location.hash !== `#${next}`) {
        window.history.replaceState(null, "", `#${next}`);
      }
      setPage(next);
      setMenuOpen(false);
    };
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);

  useEffect(() => {
    let active = true;
    Promise.allSettled([
      api.readiness(), api.signals(), api.sources(), api.boundary(),
      api.layer("verified_event"), api.layer("official_event_catalogue"),
      api.layer("observed_surveillance"), api.layer("coverage"),
    ]).then((results) => {
      if (!active) return;
      const [readyResult, signalResult, sourceResult, boundaryResult, verified, catalogue, observed, coverage] = results;
      if (readyResult.status === "fulfilled") {
        setCapabilities(Array.isArray(readyResult.value.data?.capabilities) ? readyResult.value.data.capabilities : []);
        setCapabilityStatus("loaded");
      } else {
        setCapabilityStatus("error");
      }
      if (signalResult.status === "fulfilled" && Array.isArray(signalResult.value.data)) {
        setSignals(signalResult.value.data);
        setEvidenceCoverage(signalResult.value.context.coverage_state);
      }
      if (sourceResult.status === "fulfilled" && Array.isArray(sourceResult.value.data)) setSources(sourceResult.value.data);
      if (boundaryResult.status === "fulfilled") setBoundary(boundaryResult.value);
      setLayers({
        verified_event: layerSnapshotFrom(verified),
        official_event_catalogue: layerSnapshotFrom(catalogue),
        observed_surveillance: layerSnapshotFrom(observed),
        coverage: layerSnapshotFrom(coverage),
      });
      setApiMode(readyResult.status === "fulfilled" && signalResult.status === "fulfilled" ? "live" : "fixture");
    });
    return () => { active = false; };
  }, []);

  /** district_id → canonical name, so a citation can name a place instead of an id. */
  const districtNames = useMemo(
    () => new Map(canonicalDistricts(boundary).map((district) => [district.id.toLowerCase(), district.name])),
    [boundary],
  );

  const navigate = (next: Page) => {
    setPage(next);
    setMenuOpen(false);
    if (window.location.hash !== `#${next}`) window.location.hash = next;
    window.scrollTo({ top: 0, behavior: "smooth" });
    window.requestAnimationFrame(() => document.getElementById("main")?.focus());
  };

  return (
    <div className="shell">
      <p className="bulletin">
        <AlertTriangle size={13} strokeWidth={3} aria-hidden="true" />
        Multilingual public-health evidence · source-grounded answers · district maps · bounded model outputs
      </p>

      <header className="masthead">
        <button type="button" className="brand" onClick={() => navigate("overview")}>
          <span className="brand__mark" aria-hidden="true"><Activity size={20} strokeWidth={3} /></span>
          <span className="brand__text">
            <strong>Janaswasthya</strong>
            <small lang="or">ଜନସ୍ୱାସ୍ଥ୍ୟ · Odisha evidence hub</small>
          </span>
        </button>
        <nav id="primary-nav" className={menuOpen ? "is-open" : ""} aria-label="Primary">
          {nav.map((item) => (
            <button
              key={item.id}
              type="button"
              aria-current={page === item.id ? "page" : undefined}
              className={page === item.id ? "is-on" : ""}
              onClick={() => navigate(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <p className="masthead__status">
          <span className={`dot dot--${apiMode}`} aria-hidden="true" />
          {apiMode === "live" ? "API live" : apiMode === "loading" ? "Checking" : "Fixtures"}
        </p>
        <button
          type="button"
          className="masthead__menu"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label={menuOpen ? "Close navigation" : "Open navigation"}
          aria-expanded={menuOpen}
          aria-controls="primary-nav"
        >
          {menuOpen ? <X strokeWidth={3} /> : <Menu strokeWidth={3} />}
        </button>
      </header>

      <main id="main" tabIndex={-1}>
        {page === "overview" && <Overview onNavigate={navigate} apiMode={apiMode} capabilities={capabilities} capabilityStatus={capabilityStatus} />}
        {page === "assistant" && <AgentChat districtNames={districtNames} signals={signals} />}
        {page === "translate" && <TranslationPage />}
        {page === "evidence" && (
          <Evidence
            signals={signals}
            boundary={boundary}
            apiMode={apiMode}
            evidenceCoverage={evidenceCoverage}
            layers={layers}
            capabilities={capabilities}
            capabilityStatus={capabilityStatus}
          />
        )}
        {page === "review" && <Review signals={signals} />}
        {page === "provenance" && <Provenance signals={signals} />}
        {page === "sources" && <Sources sources={sources} />}
        {page === "forecast" && <Forecast boundary={boundary} />}
      </main>

      <footer className="footer">
        <div className="footer__brand">
          <Layers size={18} strokeWidth={3} aria-hidden="true" />
          <strong>Janaswasthya</strong>
          <span>Evidence before inference.</span>
        </div>
        <p>
          English interface, bounded redacted Odia and Hindi evidence rendered in its own script ·{" "}
          <button type="button" onClick={() => navigate("sources")}>Source and licence register</button>
        </p>
        <p className="footer__build"><Grid3x3 size={12} strokeWidth={3} aria-hidden="true" /> production_shaped · schema 1.0.0</p>
      </footer>
    </div>
  );
}
