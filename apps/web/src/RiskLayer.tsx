import { useEffect, useMemo, useState } from "react";
import { CloudRain, Droplets, Thermometer, Wind } from "lucide-react";
import { api } from "./api";
import { DistrictChoropleth, DistrictTable, MapLegend, buildScale } from "./DistrictMap";
import type { DistrictDatum } from "./DistrictMap";
import { Chip, Notice, SyntheticBadge, TypedState } from "./ui";
import type {
  CurrentConditions,
  CurrentConditionsDistrict,
  GeoFeatureCollection,
  OperationalForecastReadiness,
} from "./types";

/**
 * Field names the forecasting lane may use for the district value, in priority
 * order. `suitability_percentile` is what the shipped current-conditions map
 * payload emits; the rest are accepted so a rename does not blank the map. A row
 * carrying none of them stays `null` — unmeasured — and is drawn hatched, never
 * as a zero.
 */
const VALUE_KEYS = [
  "suitability_percentile",
  "environmental_favourability",
  "favourability",
  "favourability_index",
  "environmental_index",
  "risk_index",
  "value",
] as const;

function numberAt(source: Record<string, unknown> | undefined, key: string): number | null {
  const candidate = source?.[key];
  return typeof candidate === "number" && Number.isFinite(candidate) ? candidate : null;
}

function nested(row: CurrentConditionsDistrict, key: string): Record<string, unknown> | undefined {
  const value = row[key];
  return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

export function readFavourability(row: CurrentConditionsDistrict): { value: number; key: string } | null {
  for (const key of VALUE_KEYS) {
    const flat = numberAt(row, key);
    if (flat !== null) return { value: flat, key };
    // The unflattened artefact nests the score under `suitability`.
    const deep = numberAt(nested(row, "suitability"), key);
    if (deep !== null) return { value: deep, key: `suitability.${key}` };
  }
  return null;
}

function text(row: CurrentConditionsDistrict | undefined, key: string): string | null {
  const value = row?.[key];
  return typeof value === "string" && value.trim() ? value : null;
}

function measure(row: CurrentConditionsDistrict | undefined, key: string, unit: string): string | null {
  const value = row?.[key];
  return typeof value === "number" && Number.isFinite(value) ? `${value} ${unit}` : null;
}

/**
 * One entry of `top_drivers`, which the forecasting lane emits as
 * `{feature, contribution, ...}`. Rendering the object itself printed
 * "[object Object]" on every card, which told the operator nothing about why a
 * district is shaded the way it is. A driver in an unexpected shape is skipped
 * rather than stringified.
 */
function driverLabel(driver: unknown): string {
  if (typeof driver === "string") return driver;
  const record = driver && typeof driver === "object" ? (driver as Record<string, unknown>) : null;
  const feature = typeof record?.feature === "string" ? record.feature.replaceAll("_", " ") : null;
  if (!feature) return "driver not named";
  const contribution = typeof record?.contribution === "number" ? record.contribution : null;
  if (contribution === null) return feature;
  return `${feature} ${contribution > 0 ? "+" : "−"}${Math.abs(contribution).toFixed(2)}`;
}

/**
 * IMD and climate context for one district.
 *
 * The user of this panel is asking "why is this district shaded like that", so
 * the readout leads with the India Meteorological Department fields — active
 * warning, nowcast, station rainfall — and then the four-week climate window the
 * suitability score is actually computed from. Every field renders "not
 * reported" rather than disappearing, so a gap in the IMD feed stays visible.
 */
function ConditionsReadout({ row, name }: { row: CurrentConditionsDistrict; name: string }) {
  const drivers = Array.isArray(row.top_drivers) ? (row.top_drivers as unknown[]) : [];
  const warningDays = Array.isArray(row.imd_warning_days) ? (row.imd_warning_days as unknown[]) : [];
  const band = text(row, "band");
  const imd: Array<[string, string | null]> = [
    ["Peak warning, next 5 days", text(row, "imd_peak_warning_next_5_days")],
    ["Nowcast severity", text(row, "imd_nowcast_severity")],
    ["Nowcast valid to (IST)", text(row, "imd_nowcast_valid_upto_ist")],
    ["Station", text(row, "imd_station")],
    ["Station rainfall, 24 h", measure(row, "imd_rainfall_24h_mm", "mm")],
  ];
  const climate: Array<[string, string | null, React.ReactNode]> = [
    ["Rain, 4 weeks", measure(row, "rain_4w_mm", "mm"), <Droplets key="r" size={13} strokeWidth={3} aria-hidden="true" />],
    ["Rain anomaly", measure(row, "rain_4w_anomaly_sd", "SD"), <Wind key="a" size={13} strokeWidth={3} aria-hidden="true" />],
    ["Humidity, 4 weeks", measure(row, "rh_4w_pct", "%"), <Droplets key="h" size={13} strokeWidth={3} aria-hidden="true" />],
    ["Temperature, 4 weeks", measure(row, "t2m_4w_c", "°C"), <Thermometer key="t" size={13} strokeWidth={3} aria-hidden="true" />],
    ["Longest dry run", measure(row, "longest_dry_run_4w_days", "days"), <CloudRain key="d" size={13} strokeWidth={3} aria-hidden="true" />],
  ];

  return (
    <div className="readout-card">
      <div className="readout-card__head">
        <div>
          <span className="eyebrow">District conditions</span>
          <h3>{name}</h3>
        </div>
        {band && <Chip tone="mute">{band.replaceAll("_", " ")}</Chip>}
      </div>

      <div className="readout-card__grid">
        <section>
          <span className="eyebrow">India Meteorological Department</span>
          <dl>
            {imd.map(([label, value]) => (
              <div key={label}><dt>{label}</dt><dd>{value ?? "not reported"}</dd></div>
            ))}
          </dl>
          {warningDays.length > 0 && (
            <p className="readout-card__days">
              {warningDays.length} warning {warningDays.length === 1 ? "day" : "days"} in the IMD bulletin window
            </p>
          )}
        </section>
        <section>
          <span className="eyebrow">Four-week climate window</span>
          <dl>
            {climate.map(([label, value, icon]) => (
              <div key={label}><dt>{icon} {label}</dt><dd>{value ?? "not reported"}</dd></div>
            ))}
          </dl>
          {text(row, "environment_week") && <p className="readout-card__days">Week of {text(row, "environment_week")}</p>}
        </section>
      </div>

      {drivers.length > 0 && (
        <p className="readout-card__drivers">
          <span className="eyebrow">Top drivers</span>
          {drivers.map((driver, index) => <code key={index}>{driverLabel(driver)}</code>)}
        </p>
      )}
    </div>
  );
}

/**
 * Today's environmental conditions across the 30 districts.
 *
 * The route already exists and answers with a typed refusal while the layer is
 * still being fitted, so this panel is live either way: it draws a map when the
 * payload carries district values and renders the API's own refusal when it does
 * not. It never falls back to a cached or invented surface.
 */
export function CurrentConditionsLayer({
  boundary, districts,
}: {
  boundary: GeoFeatureCollection | null;
  districts: Array<{ id: string; name: string }>;
}) {
  const [payload, setPayload] = useState<CurrentConditions | null>(null);
  const [operationalReadiness, setOperationalReadiness] = useState<OperationalForecastReadiness | null>(null);
  const [coverage, setCoverage] = useState("unknown");
  const [state, setState] = useState<"loading" | "loaded" | "error">("loading");
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<"map" | "table">("map");

  useEffect(() => {
    let active = true;
    api.currentConditions()
      .then((response) => {
        if (!active) return;
        setPayload(response.data);
        setCoverage(response.context?.coverage_state ?? "unknown");
        setState("loaded");
      })
      .catch(() => { if (active) setState("error"); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    api.operationalForecastReadiness()
      .then((response) => { if (active) setOperationalReadiness(response.data); })
      .catch(() => { if (active) setOperationalReadiness(null); });
    return () => { active = false; };
  }, []);

  const rows = useMemo(() => payload?.districts ?? [], [payload]);

  const { data, valueKey, measured } = useMemo(() => {
    const byId = new Map<string, number>();
    let key: string | null = null;
    for (const row of rows) {
      const read = readFavourability(row);
      if (!read || !row.district_id) continue;
      key = key ?? read.key;
      byId.set(row.district_id.toLowerCase(), read.value);
    }
    const datums: DistrictDatum[] = districts.map((district) => ({
      districtId: district.id,
      name: district.name,
      value: byId.has(district.id.toLowerCase()) ? byId.get(district.id.toLowerCase()) ?? null : null,
    }));
    return { data: datums, valueKey: key, measured: byId.size };
  }, [districts, rows]);

  const { bins, lowers } = useMemo(() => buildScale(data.map((entry) => entry.value)), [data]);
  const noDataCount = data.filter((entry) => entry.value === null).length;
  const publishes = state === "loaded" && measured > 0;
  const metric = payload?.metric
    ?? (valueKey?.includes("suitability_percentile") ? "Environmental suitability percentile" : null)
    ?? (valueKey ? valueKey.replaceAll("_", " ") : "Environmental favourability");

  const statement = payload?.quantity_statement
    ?? (payload?.quantity && typeof payload.quantity === "object"
      ? (payload.quantity as Record<string, unknown>).statement
      : null);

  const dataEdge = payload?.data_edge;
  const selectedRow = rows.find((row) => String(row.district_id).toLowerCase() === selected?.toLowerCase());
  const selectedName = districts.find((district) => district.id === selected)?.name ?? selected ?? "";

  return (
    <div className="conditions">
      <div className="conditions__head">
        <div>
          <span className="eyebrow"><CloudRain size={13} strokeWidth={3} aria-hidden="true" /> Current risk factors · 30 districts</span>
          <h2>Environmental early-warning map</h2>
        </div>
        {payload?.is_synthetic
          ? <SyntheticBadge label="Synthetic layer" />
          : <Chip tone={publishes ? "ok" : "mute"}>{coverage.replaceAll("_", " ")}</Chip>}
      </div>

      {state === "loading" && (
        <p className="pending pending--block">
          Loading the district environmental layer. Nothing is drawn until the API answers.
        </p>
      )}

      {state === "error" && <TypedState code="unknown" capability="current_conditions_layer" compact />}

      {state === "loaded" && payload && (
        <>
          <p className="conditions__quantity">
            {typeof statement === "string"
              ? statement
              : "Relative environmental risk-factor index derived from observed rainfall, temperature and humidity."}
          </p>

          {!publishes && (
            <>
              <TypedState
                code={payload.capability_code ?? payload.status ?? "insufficient_evidence"}
                capability="current_conditions_layer"
                detail={payload.message ?? undefined}
              />
              {payload.unlocked_by && (
                <p className="conditions__unlock"><strong>Unlocked by</strong> {payload.unlocked_by}</p>
              )}
              {(payload.reason_codes ?? []).length > 0 && (
                <div className="refusal__codes">
                  {(payload.reason_codes ?? []).map((code) => <code key={code}>{code}</code>)}
                </div>
              )}
            </>
          )}

          {publishes && (
            <>
              <Notice tone="ok" title="Decision-support interpretation">
                <p>
                  Higher values prioritise districts whose current weather is unusually favourable relative to their
                  own history. Combine this layer with recent source evidence and human verification before escalation.
                </p>
              </Notice>

              {operationalReadiness && (
                <Notice
                  tone={operationalReadiness.eligible_for_training ? "ok" : "warn"}
                  title="Disease-probability model readiness"
                >
                  {operationalReadiness.eligible_for_training ? (
                    <p>
                      The authorised district-week data structure has passed its initial history and completeness
                      gate. Disease-specific rolling-origin backtests and calibration remain required before a
                      probability is released.
                    </p>
                  ) : (
                    <p>
                      This live map remains the early-warning layer while the authorised aggregate district-week
                      case export is pending. The ready path requires cases, explicit weekly NIL reports, reporting
                      completeness, population and data-vintage dates—never patient line lists.
                    </p>
                  )}
                </Notice>
              )}

              <div className="conditions__bar">
                <p>
                  <strong>{measured}</strong> of {data.length} districts carry a value for{" "}
                  <code>{valueKey ?? "unnamed field"}</code>
                  {payload.as_of ? ` as of ${payload.as_of}` : payload.generated_at ? ` as of ${payload.generated_at.slice(0, 10)}` : ""}.
                </p>
                <div className="toggle" role="group" aria-label="Map or table view">
                  <button type="button" className={`toggle__btn${view === "map" ? " toggle__btn--on" : ""}`} aria-pressed={view === "map"} onClick={() => setView("map")}>Map</button>
                  <button type="button" className={`toggle__btn${view === "table" ? " toggle__btn--on" : ""}`} aria-pressed={view === "table"} onClick={() => setView("table")}>Table</button>
                </div>
              </div>

              {view === "map" && (
                <DistrictChoropleth
                  boundary={boundary}
                  data={data}
                  bins={bins}
                  lowers={lowers}
                  metric={metric}
                  describedBy="conditions-table"
                  highlighted={highlighted}
                  onHighlight={setHighlighted}
                  onSelect={setSelected}
                  selected={selected}
                />
              )}
              <DistrictTable
                id="conditions-table"
                data={data}
                bins={bins}
                lowers={lowers}
                metric={metric}
                unit={metric}
                hidden={view === "map"}
                highlighted={highlighted}
                onHighlight={setHighlighted}
                onSelect={setSelected}
                selected={selected}
              />
              <MapLegend bins={bins} metric={metric} noDataCount={noDataCount} totalCount={data.length} />

              {selectedRow
                ? <ConditionsReadout row={selectedRow} name={selectedName} />
                : <p className="readout">Select a district to read its IMD warning, nowcast and four-week climate window.</p>}

              {dataEdge && (
                <dl className="kv kv--4 conditions__edge">
                  <div><dt>Climate observed to</dt><dd>{String(dataEdge.nasa_power_last_observed_day ?? "not reported")}</dd></div>
                  <div><dt>Climate window</dt><dd>{String(dataEdge.nasa_power_window ?? "not reported")}</dd></div>
                  <div><dt>IMD collected at</dt><dd>{String(dataEdge.imd_collected_at ?? "not reported")}</dd></div>
                  <div><dt>Districts scored</dt><dd>{measured} of {data.length}</dd></div>
                </dl>
              )}

              {(payload.warnings ?? []).map((warning, index) => (
                <p key={index} className="conditions__warning">{String(warning)}</p>
              ))}
              {payload.warning && <p className="conditions__warning">{payload.warning}</p>}
            </>
          )}
        </>
      )}
    </div>
  );
}
