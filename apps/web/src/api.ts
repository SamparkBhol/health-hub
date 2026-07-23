import type {
  AgentQueryRequest,
  AgentQueryResult,
  CurrentConditions,
  Envelope,
  EvidenceLayerRecord,
  GeoFeatureCollection,
  LayerType,
  OperationalForecastReadiness,
  OperationalForecastMap,
  OperationalForecastSummary,
  PublishedSignalMap,
  PublicHmisMap,
  PublicMalariaMap,
  PublicOutlook,
  ReadinessData,
  RealForecastMap,
  RealForecastSummary,
  Signal,
  SignalFilters,
  SourceState,
  SyntheticForecastReport,
  TranslateRequest,
  TranslationResult,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export class ApiUnavailable extends Error {
  /** HTTP status when the server answered; undefined for a transport failure. */
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(path: string, timeoutMs = 5000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!response.ok) throw new ApiUnavailable(`${response.status} ${path}`, response.status);
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiUnavailable) throw error;
    throw new ApiUnavailable(error instanceof Error ? error.message : "API unavailable");
  } finally {
    window.clearTimeout(timer);
  }
}

async function getStaticJson<T>(path: string, timeoutMs = 5000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!response.ok) throw new ApiUnavailable(`${response.status} ${path}`);
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiUnavailable) throw error;
    throw new ApiUnavailable(error instanceof Error ? error.message : "Static asset unavailable");
  } finally {
    window.clearTimeout(timer);
  }
}

async function postJson<T>(path: string, body: unknown, timeoutMs = 15000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) throw new ApiUnavailable(`${response.status} ${path}`, response.status);
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiUnavailable) throw error;
    throw new ApiUnavailable(error instanceof Error ? error.message : "API unavailable");
  } finally {
    window.clearTimeout(timer);
  }
}

function queryString(values: Record<string, string | number | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}

export const api = {
  readiness: () => getJson<Envelope<ReadinessData>>("/api/v1/readiness"),
  sources: () => getJson<Envelope<SourceState[]>>("/api/v1/sources"),
  signals: (filters: SignalFilters = {}) =>
    getJson<Envelope<Signal[]>>(`/api/v1/signals${queryString({ ...filters, limit: 200 })}`),
  publishedSignalMap: (filters: SignalFilters = {}) =>
    getJson<Envelope<PublishedSignalMap>>(
      `/api/v1/maps/published-signals${queryString({
        disease: filters.disease,
        language: filters.language,
        assertion: filters.assertion,
        retrieved_from: filters.retrieved_from,
        retrieved_to: filters.retrieved_to,
      })}`,
    ),
  publicMalariaMap: (year?: number, metric = "api") =>
    getJson<Envelope<PublicMalariaMap>>(
      `/api/v1/public-health/malaria/map${queryString({ year, metric })}`,
      15000,
    ),
  /** District-monthly HMIS panel. `period` is a month-start such as `2020-03-01`. */
  publicHmisMap: (period?: string, metric = "malaria_test_positivity") =>
    getJson<Envelope<PublicHmisMap>>(
      `/api/v1/public-health/hmis/map${queryString({ period, metric })}`,
      15000,
    ),
  publicOutlook: (horizonMonth: 1 | 2 | 3) =>
    getJson<Envelope<PublicOutlook>>(
      `/api/v1/outlook/public/map${queryString({ disease: "malaria", horizon_month: horizonMonth })}`,
      15000,
    ),
  syntheticForecast: (horizonWeeks: 1 | 2 | 4 | 8 | 12 = 12) =>
    getJson<Envelope<SyntheticForecastReport>>(
      `/api/v1/demo/synthetic-forecast${queryString({ horizon_weeks: horizonWeeks })}`,
      15000,
    ),
  // The real reported-outbreak occurrence model. Distinct from `syntheticForecast`
  // (a software harness) and from `/api/v1/forecast` (incidence, still refused).
  realForecastSummary: () =>
    getJson<Envelope<RealForecastSummary>>("/api/v1/forecast/real", 15000),
  realForecastMap: (diseaseGroup: string, horizonWeeks: number) =>
    getJson<Envelope<RealForecastMap>>(
      `/api/v1/forecast/real/map${queryString({
        disease_group: diseaseGroup,
        horizon_weeks: horizonWeeks,
      })}`,
      15000,
    ),
  /** Today's district environmental-conditions map, separate from disease risk. */
  currentConditions: () =>
    getJson<Envelope<CurrentConditions>>("/api/v1/environment/current/map", 15000),
  operationalForecastReadiness: () =>
    getJson<Envelope<OperationalForecastReadiness>>("/api/v1/forecast/operational/readiness"),
  operationalForecastSummary: () =>
    getJson<Envelope<OperationalForecastSummary>>("/api/v1/forecast/operational", 15000),
  operationalForecastMap: (disease: string, horizonWeeks: number) =>
    getJson<Envelope<OperationalForecastMap>>(
      `/api/v1/forecast/operational/map${queryString({ disease, horizon_weeks: horizonWeeks })}`,
      15000,
    ),
  layer: (layer: LayerType) => getJson<Envelope<EvidenceLayerRecord[]>>(`/api/v1/layers/${layer}`),
  audit: () => getJson<Envelope<Record<string, unknown>>>("/api/v1/audits/epiclim"),
  rawAgentQuery: (request: AgentQueryRequest) =>
    postJson<Envelope<AgentQueryResult>>("/api/v1/agent/query", request, 120000),
  agentQuery: (request: AgentQueryRequest) =>
    postJson<Envelope<AgentQueryResult>>("/api/v1/agent/query", request, 120000),
  translate: (request: TranslateRequest) =>
    postJson<Envelope<TranslationResult>>("/api/v1/translate", request, 120000),
  boundary: async () => {
    const configured = (import.meta.env.VITE_BOUNDARY_URL as string | undefined) ?? "";
    const candidates: Array<[string, "static" | "api"]> = [
      ...(configured ? [[configured, "static"]] as Array<[string, "static"]> : []),
      ["/api/v1/boundaries/districts", "api"],
      ["/data/odisha_districts_census_2011.geojson", "static"],
    ];
    for (const [candidate, source] of candidates) {
      try {
        return source === "api"
          ? await getJson<GeoFeatureCollection>(candidate)
          : await getStaticJson<GeoFeatureCollection>(candidate);
      } catch {
        // The UI has an explicitly labelled non-geographic table fallback.
      }
    }
    throw new ApiUnavailable("District geometry unavailable");
  },
};
