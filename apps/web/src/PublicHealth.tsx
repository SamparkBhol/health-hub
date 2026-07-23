import { useEffect, useMemo, useState } from "react";
import { ExternalLink, Map as MapIcon, Table2 } from "lucide-react";
import { api } from "./api";
import { DistrictChoropleth, DistrictTable, MapLegend, buildScale } from "./DistrictMap";
import type { DistrictDatum } from "./DistrictMap";
import {
  HMIS_EPISTEMIC_CAPTION, HMIS_METRICS, formatHmisPeriod, hmisMetricMeta, hmisPeriods, scaleHmisValue,
} from "./hmis";
import { MALARIA_METRICS, malariaMetricMeta } from "./publicHealthMeta";
import { Chip, Notice, TypedState, WarningStrip } from "./ui";
import type {
  GeoFeatureCollection, PublicHmisMap, PublicMalariaMap, PublicOutlook, Warning,
} from "./types";

interface District {
  id: string;
  name: string;
}

/**
 * A map that keeps its last good payload on screen while the next one loads.
 *
 * Blanking the workspace to a one-line "loading" on every selector change
 * destroys the comparison the operator is in the middle of making — they change
 * the year precisely to see what moved. The previous surface therefore stays
 * rendered, dimmed and marked stale, until the replacement arrives.
 */
function StaleBar({ stale, label }: { stale: boolean; label: string }) {
  if (!stale) return null;
  return (
    <p className="stale-bar" role="status">
      <span className="stale-bar__dot" aria-hidden="true" />
      {label} The values below are the previous selection until it answers.
    </p>
  );
}

export function OfficialMalariaMap({
  boundary, districts,
}: {
  boundary: GeoFeatureCollection | null;
  districts: District[];
}) {
  const [payload, setPayload] = useState<PublicMalariaMap | null>(null);
  const [warnings, setWarnings] = useState<Warning[]>([]);
  const [metric, setMetric] = useState("api");
  // `null` means "whatever year the API considers latest". Storing the response's
  // own year here would re-fire the effect and refetch the map it just drew.
  const [year, setYear] = useState<number | null>(null);
  const [state, setState] = useState<"loading" | "loaded" | "error">("loading");
  const [view, setView] = useState<"map" | "table">("map");
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setState("loading");
    api.publicMalariaMap(year ?? undefined, metric)
      .then((response) => {
        if (!active) return;
        setPayload(response.data);
        setWarnings(response.warnings ?? []);
        setState("loaded");
      })
      .catch(() => { if (active) setState("error"); });
    return () => { active = false; };
  }, [metric, year]);

  const byId = useMemo(
    () => new Map((payload?.records ?? []).map((row) => [row.district_id, row])),
    [payload],
  );
  const data = useMemo<DistrictDatum[]>(() => districts.map((district) => ({
    districtId: district.id,
    name: district.name,
    value: byId.get(district.id)?.value ?? null,
  })), [byId, districts]);
  const { bins, lowers } = useMemo(() => buildScale(data.map((item) => item.value)), [data]);
  const selectedRow = selected ? byId.get(selected) : null;
  // The framing follows the metric: ABER is testing effort and Pf% is parasite
  // mix, and neither is disease burden however the map is coloured.
  const meta = malariaMetricMeta(payload?.metric ?? metric);
  const stale = state === "loading" && payload !== null;

  return (
    <section className="conditions" aria-labelledby="official-malaria-heading">
      <div className="conditions__head">
        <div>
          <span className="eyebrow">{meta.eyebrow}</span>
          <h2 id="official-malaria-heading">NCVBDC district malaria heatmap</h2>
        </div>
        <Chip tone={meta.framing === "burden" ? "ok" : "warn"}>
          {meta.framing === "burden" ? "30 / 30 districts · burden" : "30 / 30 districts · not burden"}
        </Chip>
      </div>
      <p className="conditions__quantity">
        {payload?.metric_definition ?? "Loading the official annual district malaria table…"}
      </p>
      <p className="conditions__scope">{meta.caption}</p>
      <div className="filters" role="group" aria-label="Official malaria map controls">
        <label><span>Year</span><select
          value={year ?? payload?.year ?? ""}
          onChange={(event) => setYear(Number(event.target.value))}
        >
          {(payload?.available_years ?? [2024]).map((value) => <option key={value} value={value}>{value}</option>)}
        </select></label>
        <label><span>Metric</span><select value={metric} onChange={(event) => setMetric(event.target.value)}>
          {MALARIA_METRICS.map((entry) => (
            <option key={entry.value} value={entry.value}>{entry.label}</option>
          ))}
        </select></label>
        <div className="toggle" role="group" aria-label="Map or table view">
          <button type="button" className={`toggle__btn${view === "map" ? " toggle__btn--on" : ""}`} onClick={() => setView("map")}><MapIcon size={16} /> Map</button>
          <button type="button" className={`toggle__btn${view === "table" ? " toggle__btn--on" : ""}`} onClick={() => setView("table")}><Table2 size={16} /> Table</button>
        </div>
      </div>
      <WarningStrip warnings={warnings} />
      <StaleBar stale={stale} label="Reading the bundled, hash-pinned NCVBDC table." />
      {state === "loading" && !payload && (
        <p className="pending pending--block">Reading the bundled, hash-pinned NCVBDC table…</p>
      )}
      {state === "error" && <TypedState code="source_temporarily_unavailable" capability="official_public_malaria_map" compact />}
      {payload && state !== "error" && (
        <div className={`workspace${stale ? " workspace--stale" : ""}`}>
          <div className="workspace__map">
            {view === "map" && <DistrictChoropleth boundary={boundary} data={data} bins={bins} lowers={lowers} metric={payload.metric_definition} describedBy="official-malaria-table" highlighted={highlighted} onHighlight={setHighlighted} onSelect={setSelected} selected={selected} />}
            <DistrictTable id="official-malaria-table" data={data} bins={bins} lowers={lowers} metric={payload.metric_definition} unit={meta.unit} hidden={view === "map"} highlighted={highlighted} onHighlight={setHighlighted} onSelect={setSelected} selected={selected} />
            <MapLegend bins={bins} metric={payload.metric_definition} noDataCount={data.filter((row) => row.value === null).length} totalCount={30} />
          </div>
          <aside className="workspace__side">
            {selectedRow ? (
              <article className="readout-card">
                <span className="eyebrow">Official {selectedRow.year} row</span>
                <h3>{selectedRow.district_name}</h3>
                <dl>
                  <div><dt>Annual Parasite Incidence</dt><dd>{selectedRow.api}</dd></div>
                  <div><dt>Cases / positives</dt><dd>{selectedRow.total_cases ?? "not printed"}</dd></div>
                  <div><dt>Reported deaths</dt><dd>{selectedRow.deaths ?? "not printed"}</dd></div>
                  <div><dt>Slide Positivity Rate</dt><dd>{selectedRow.spr}</dd></div>
                  <div><dt>Annual Blood Examination Rate</dt><dd>{selectedRow.aber}</dd></div>
                  <div><dt>P. falciparum</dt><dd>{selectedRow.pf_percent}%</dd></div>
                </dl>
                <a href={selectedRow.source_url} target="_blank" rel="noreferrer">Open official report <ExternalLink size={13} /></a>
              </article>
            ) : (
              <Notice tone="ok" title="This is disease data, not article density">
                <p>Select a district to inspect its complete official row and source report. The latest bundled table is annual 2024, so it is not presented as current-week incidence.</p>
              </Notice>
            )}
          </aside>
        </div>
      )}
    </section>
  );
}

/**
 * The district-monthly HMIS panel.
 *
 * This is the only surface in the product with a monthly time axis, which makes
 * it the easiest one to misread as incidence. Every value here is a count of
 * facility-submitted records: no denominator, no de-duplication of people, and
 * provisional until later submissions revise it. The caption states that beside
 * the map rather than in a footnote.
 */
export function OfficialHmisMap({
  boundary, districts,
}: {
  boundary: GeoFeatureCollection | null;
  districts: District[];
}) {
  const [payload, setPayload] = useState<PublicHmisMap | null>(null);
  const [warnings, setWarnings] = useState<Warning[]>([]);
  const [metric, setMetric] = useState("malaria_test_positivity");
  /** `null` means the latest month the panel holds. */
  const [period, setPeriod] = useState<string | null>(null);
  const [state, setState] = useState<"loading" | "loaded" | "error">("loading");
  const [view, setView] = useState<"map" | "table">("map");
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setState("loading");
    api.publicHmisMap(period ?? undefined, metric)
      .then((response) => {
        if (!active) return;
        setPayload(response.data);
        setWarnings(response.warnings ?? []);
        setState("loaded");
      })
      .catch(() => { if (active) setState("error"); });
    return () => { active = false; };
  }, [metric, period]);

  const byId = useMemo(
    () => new Map((payload?.records ?? []).map((row) => [row.district_id, row])),
    [payload],
  );
  const meta = hmisMetricMeta(payload?.metric ?? metric);
  const data = useMemo<DistrictDatum[]>(() => districts.map((district) => ({
    districtId: district.id,
    name: district.name,
    value: scaleHmisValue(byId.get(district.id)?.value, meta),
  })), [byId, districts, meta]);
  const { bins, lowers } = useMemo(() => buildScale(data.map((item) => item.value)), [data]);
  const periods = useMemo(
    () => hmisPeriods(payload?.available_period_start, payload?.available_period_end),
    [payload?.available_period_end, payload?.available_period_start],
  );
  const selectedRow = selected ? byId.get(selected) : null;
  const stale = state === "loading" && payload !== null;
  const shownPeriod = payload ? formatHmisPeriod(payload.period) : "";
  // The unit travels with the metric name so the numerals drawn on the map are
  // never read as a bare count of people.
  const mapMetric = `${meta.label} · ${meta.unit} · ${shownPeriod}`;

  return (
    <section className="conditions" aria-labelledby="official-hmis-heading">
      <div className="conditions__head">
        <div>
          <span className="eyebrow">Facility-reported records · not incidence</span>
          <h2 id="official-hmis-heading">HMIS district-monthly reporting panel</h2>
        </div>
        <Chip tone="warn">provisional records</Chip>
      </div>
      <p className="conditions__quantity">
        {payload ? `${meta.label} · ${shownPeriod}. ${meta.definition}` : "Loading the bundled Odisha HMIS monthly panel…"}
      </p>
      <p className="conditions__scope">{payload?.metric_scope ?? HMIS_EPISTEMIC_CAPTION}</p>
      <div className="filters" role="group" aria-label="HMIS map controls">
        <label><span>Reporting month</span><select
          value={period ?? payload?.period ?? ""}
          onChange={(event) => setPeriod(event.target.value)}
          disabled={!periods.length}
        >
          {(periods.length ? periods : payload ? [payload.period] : []).map((value) => (
            <option key={value} value={value}>{formatHmisPeriod(value)}</option>
          ))}
        </select></label>
        <label><span>Metric</span><select value={metric} onChange={(event) => setMetric(event.target.value)}>
          {HMIS_METRICS.map((entry) => (
            <option key={entry.value} value={entry.value}>{entry.label}</option>
          ))}
        </select></label>
        <div className="toggle" role="group" aria-label="Map or table view">
          <button type="button" className={`toggle__btn${view === "map" ? " toggle__btn--on" : ""}`} onClick={() => setView("map")}><MapIcon size={16} /> Map</button>
          <button type="button" className={`toggle__btn${view === "table" ? " toggle__btn--on" : ""}`} onClick={() => setView("table")}><Table2 size={16} /> Table</button>
        </div>
      </div>
      <WarningStrip warnings={warnings} />
      <StaleBar stale={stale} label="Reading the bundled Odisha HMIS monthly panel." />
      {state === "loading" && !payload && (
        <p className="pending pending--block">Reading the bundled Odisha HMIS monthly panel…</p>
      )}
      {state === "error" && (
        <TypedState code="source_temporarily_unavailable" capability="official_public_hmis_map" compact />
      )}
      {payload && state !== "error" && (
        <div className={`workspace${stale ? " workspace--stale" : ""}`}>
          <div className="workspace__map">
            {view === "map" && (
              <DistrictChoropleth
                boundary={boundary}
                data={data}
                bins={bins}
                lowers={lowers}
                metric={mapMetric}
                describedBy="official-hmis-table"
                highlighted={highlighted}
                onHighlight={setHighlighted}
                onSelect={setSelected}
                selected={selected}
              />
            )}
            <DistrictTable
              id="official-hmis-table"
              data={data}
              bins={bins}
              lowers={lowers}
              metric={mapMetric}
              unit={meta.unit}
              hidden={view === "map"}
              highlighted={highlighted}
              onHighlight={setHighlighted}
              onSelect={setSelected}
              selected={selected}
            />
            <MapLegend
              bins={bins}
              metric={mapMetric}
              noDataCount={data.filter((row) => row.value === null).length}
              totalCount={data.length}
            />
          </div>
          <aside className="workspace__side">
            {selectedRow ? (
              <article className="readout-card">
                <span className="eyebrow">HMIS {shownPeriod} · facility records</span>
                <h3>{selectedRow.district_name}</h3>
                <dl>
                  <div><dt>Malaria tests</dt><dd>{selectedRow.malaria_tests ?? "not reported"}</dd></div>
                  <div><dt>Malaria positive records</dt><dd>{selectedRow.malaria_positive_records ?? "not reported"}</dd></div>
                  <div><dt>Microscopy slides</dt><dd>{selectedRow.malaria_microscopy_tests ?? "not reported"}</dd></div>
                  <div><dt>Microscopy positive records</dt><dd>{selectedRow.malaria_microscopy_positive_records ?? "not reported"}</dd></div>
                  <div><dt>Dengue positive records</dt><dd>{selectedRow.dengue_positive_records ?? "not reported"}</dd></div>
                  <div><dt>Childhood diarrhoea records</dt><dd>{selectedRow.childhood_diarrhoea_records ?? "not reported"}</dd></div>
                  <div><dt>Observation state</dt><dd>{selectedRow.observation_state.replaceAll("_", " ")}</dd></div>
                </dl>
                <a href={selectedRow.resource_url ?? selectedRow.source_url} target="_blank" rel="noreferrer">
                  Open the published HMIS resource <ExternalLink size={13} />
                </a>
              </article>
            ) : (
              <Notice tone="warn" title="Records reported, not people counted">
                <p>{HMIS_EPISTEMIC_CAPTION}</p>
                <p>
                  Select a district to read the raw monthly counts behind its shade, including the test
                  denominators the positivity metrics are computed from.
                </p>
              </Notice>
            )}
          </aside>
        </div>
      )}
    </section>
  );
}

export function PublicResearchOutlook({
  boundary, districts,
}: {
  boundary: GeoFeatureCollection | null;
  districts: District[];
}) {
  const [horizon, setHorizon] = useState<1 | 2 | 3>(1);
  const [payload, setPayload] = useState<PublicOutlook | null>(null);
  const [warnings, setWarnings] = useState<Warning[]>([]);
  const [state, setState] = useState<"loading" | "loaded" | "error">("loading");
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setState("loading");
    api.publicOutlook(horizon)
      .then((response) => {
        if (!active) return;
        setPayload(response.data);
        setWarnings(response.warnings ?? []);
        setState("loaded");
      })
      .catch(() => { if (active) setState("error"); });
    return () => { active = false; };
  }, [horizon]);

  const byId = useMemo(
    () => new Map((payload?.records ?? []).map((row) => [row.district_id, row])),
    [payload],
  );
  const data = useMemo<DistrictDatum[]>(() => districts.map((district) => ({
    districtId: district.id,
    name: district.name,
    value: byId.get(district.id)?.surveillance_priority_score ?? null,
  })), [byId, districts]);
  const { bins, lowers } = useMemo(() => buildScale(data.map((item) => item.value)), [data]);
  const selectedRow = selected ? byId.get(selected) : null;
  const ridge = payload?.model_evaluation.ridge_logistic;
  const baseline = payload?.model_evaluation.calendar_month_baseline;
  const stale = state === "loading" && payload !== null;

  return (
    <section className="conditions" aria-labelledby="public-outlook-heading">
      <div className="conditions__head">
        <div><span className="eyebrow">1–3 month predictive analysis · public data</span><h2 id="public-outlook-heading">Malaria surveillance-priority outlook</h2></div>
        <Chip tone="warn">research, not operational alert</Chip>
      </div>
      <label className="filters__label"><span>Lead window</span><select value={horizon} onChange={(event) => setHorizon(Number(event.target.value) as 1 | 2 | 3)}>
        <option value={1}>Month 1 · days 1–30</option><option value={2}>Month 2 · days 31–60</option><option value={3}>Month 3 · days 61–90</option>
      </select></label>
      <WarningStrip warnings={warnings} />
      <StaleBar stale={stale} label="Combining the public health history with the 51-member seasonal ensemble." />
      {state === "loading" && !payload && <p className="pending pending--block">Combining the public health history with the 51-member seasonal ensemble…</p>}
      {state === "error" && <TypedState code="source_temporarily_unavailable" capability="public_three_month_outlook" compact />}
      {payload && state !== "error" && (
        <>
          <p className="conditions__quantity">{payload.priority_definition}</p>
          <div className="grid grid--3">
            <article className="slab requirement"><strong>Target</strong><p>{payload.forecast_target}</p></article>
            <article className="slab requirement"><strong>Model selected out of time</strong><p>{payload.selected_model.replaceAll("_", " ")}. Environment: {payload.environment_ablation_result.replaceAll("_", " ")}.</p><p className="conditions__quantity">{payload.skill_attribution}</p></article>
            <article className="slab requirement"><strong>Rolling-origin Brier</strong><p>Calendar baseline {String(baseline?.brier ?? "—")} · regularised logistic {String(ridge?.brier ?? "—")}. Lower is better.</p></article>
          </div>
          <div className={`workspace${stale ? " workspace--stale" : ""}`}>
            <div className="workspace__map">
              <DistrictChoropleth boundary={boundary} data={data} bins={bins} lowers={lowers} metric="Surveillance-priority score (not probability)" describedBy="public-outlook-table" highlighted={highlighted} onHighlight={setHighlighted} onSelect={setSelected} selected={selected} />
              <DistrictTable id="public-outlook-table" data={data} bins={bins} lowers={lowers} metric="Surveillance-priority score" unit="Priority score / 100" hidden highlighted={highlighted} onHighlight={setHighlighted} onSelect={setSelected} selected={selected} />
              <MapLegend bins={bins} metric="Surveillance-priority score" noDataCount={data.filter((row) => row.value === null).length} totalCount={30} />
            </div>
            <aside className="workspace__side">
              {selectedRow ? <article className="readout-card">
                <span className="eyebrow">{selectedRow.target_start} to {selectedRow.target_end}</span><h3>{selectedRow.district_name}</h3>
                <dl>
                  <div><dt>Planning priority</dt><dd>{selectedRow.surveillance_priority_score}/100</dd></div>
                  <div><dt>Research HMIS-indicator likelihood</dt><dd>{(selectedRow.research_indicator_probability * 100).toFixed(1)}%</dd></div>
                  <div><dt>Official {selectedRow.official_burden_year} malaria API</dt><dd>{selectedRow.official_malaria_api}</dd></div>
                  <div><dt>Forecast rain, ensemble mean</dt><dd>{selectedRow.forecast_precipitation_mean_mm} mm</dd></div>
                  <div><dt>Rain uncertainty (p10–p90)</dt><dd>{selectedRow.forecast_precipitation_p10_mm}–{selectedRow.forecast_precipitation_p90_mm} mm</dd></div>
                  <div><dt>Forecast temperature</dt><dd>{selectedRow.forecast_temperature_mean_c} °C</dd></div>
                </dl>
                <p>{selectedRow.environment_used_in_probability ? "Environment cleared ablation and enters the indicator model." : "Forecast environment is shown as context; it did not improve rolling-origin Brier enough to replace the seasonal baseline."}</p>
              </article> : <Notice tone="warn" title="Select a district for the explanation"><p>The map combines latest official annual burden and historical HMIS seasonality. The 51-member EC46/SEAS5 forecast is retained with uncertainty, but only enters the probability if it beats the seasonal baseline.</p></Notice>}
            </aside>
          </div>
          <p className="conditions__quantity"><strong>Calibration boundary:</strong> {payload.forecast_error_note}</p>
        </>
      )}
    </section>
  );
}
