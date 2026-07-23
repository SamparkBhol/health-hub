import { useMemo } from "react";
import { normalizeName } from "./districts";
import type { GeoFeatureCollection } from "./types";

// District-name resolution lives in `districts.ts` so the rule the map paints by
// and the rule the district x disease grid filters by can never drift apart.
// Re-exported here because this module was its original home.
export { normalizeName };

/**
 * One row per Odisha district, always. A district the API did not return is carried
 * through with `value: null` and is drawn as hatched "no data" — never as zero, and
 * never omitted from the map or the table.
 */
export interface DistrictDatum {
  districtId: string;
  name: string;
  value: number | null;
}

export interface ScaleBin {
  index: number;
  min: number;
  max: number;
  color: string;
  label: string;
  /** True when the fill is dark enough that a white numeral is the AA-passing choice. */
  onDark: boolean;
}

/**
 * Sequential light-blue to deep-purple ramp. Lightness decreases monotonically so the
 * order survives greyscale printing; the numeral drawn on each district and the bin
 * number in the legend carry the same information without relying on colour at all.
 */
export const RAMP = ["#B3E3FF", "#7CC4F2", "#5E93E6", "#6E56D6", "#43209B"] as const;

const NAME_KEYS = ["canonical_name", "DISTRICT", "district", "DISTRICT_1", "dtname", "shapeName", "NAME_2"];

function featureName(properties: Record<string, unknown>): string {
  const raw = NAME_KEYS.map((key) => properties[key]).find((value) => typeof value === "string" && value.trim());
  return typeof raw === "string" ? raw : "Unknown district";
}

function featureId(properties: Record<string, unknown>): string | null {
  const raw = properties.district_id;
  return typeof raw === "string" && raw.trim() ? raw.trim() : null;
}

function relativeLuminance(hex: string): number {
  const channel = (start: number) => {
    const value = parseInt(hex.slice(start, start + 2), 16) / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}

function formatValue(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.?0+$/, "");
}

/**
 * Which ramp stops a k-class scheme uses. A two-class scheme deliberately skips the
 * extreme ends: with counts of 1 and 2 an end-to-end jump reads as a dramatic gap that
 * two retrieved documents do not justify.
 */
const RAMP_SUBSETS: Record<number, number[]> = {
  1: [3],
  2: [1, 3],
  3: [0, 2, 4],
  4: [0, 1, 3, 4],
  5: [0, 1, 2, 3, 4],
};

function pickColors(count: number): string[] {
  return (RAMP_SUBSETS[count] ?? RAMP_SUBSETS[5]).map((index) => RAMP[index]);
}

export function binIndexOf(value: number, lowers: number[]): number {
  let index = 0;
  for (let i = 0; i < lowers.length; i += 1) if (value >= lowers[i]) index = i;
  return index;
}

/**
 * Builds up to five bins. When the layer only produced a handful of distinct counts —
 * the normal case for retrieved-document counts — every distinct value gets its own
 * bin so the legend reads "1", "2", "3" rather than an invented interval.
 */
export function buildScale(values: Array<number | null>): { bins: ScaleBin[]; lowers: number[] } {
  const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!finite.length) return { bins: [], lowers: [] };

  const unique = [...new Set(finite)].sort((left, right) => left - right);
  const binCount = Math.min(RAMP.length, unique.length);
  const lowers: number[] = [];
  for (let index = 0; index < binCount; index += 1) {
    const candidate = unique[Math.floor((index * unique.length) / binCount)];
    if (!lowers.length || candidate > lowers[lowers.length - 1]) lowers.push(candidate);
  }

  const colors = pickColors(lowers.length);
  const buckets: number[][] = lowers.map(() => []);
  for (const value of finite) buckets[binIndexOf(value, lowers)].push(value);

  const bins = lowers.map((lower, index) => {
    const present = buckets[index];
    const min = present.length ? Math.min(...present) : lower;
    const max = present.length ? Math.max(...present) : lower;
    const color = colors[index];
    return {
      index,
      min,
      max,
      color,
      label: min === max ? formatValue(min) : `${formatValue(min)}–${formatValue(max)}`,
      onDark: relativeLuminance(color) < 0.2,
    };
  });
  return { bins, lowers };
}

type Point = [number, number];

function polygonsOf(geometry: GeoFeatureCollection["features"][number]["geometry"]): Point[][][] {
  if (geometry.type === "Polygon") return [geometry.coordinates as Point[][]];
  return geometry.coordinates as Point[][][];
}

function ringCentroid(ring: Point[]): { x: number; y: number; area: number } {
  let twiceArea = 0;
  let x = 0;
  let y = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const cross = ring[j][0] * ring[i][1] - ring[i][0] * ring[j][1];
    twiceArea += cross;
    x += (ring[j][0] + ring[i][0]) * cross;
    y += (ring[j][1] + ring[i][1]) * cross;
  }
  if (Math.abs(twiceArea) < 1e-9) {
    const mean = ring.reduce((sum, point) => [sum[0] + point[0], sum[1] + point[1]] as Point, [0, 0] as Point);
    return { x: mean[0] / ring.length, y: mean[1] / ring.length, area: 0 };
  }
  return { x: x / (3 * twiceArea), y: y / (3 * twiceArea), area: Math.abs(twiceArea / 2) };
}

const VIEW_WIDTH = 660;
const VIEW_HEIGHT = 560;
const PADDING = 16;

interface Shape {
  key: string;
  districtId: string | null;
  name: string;
  path: string;
  labelX: number;
  labelY: number;
  labelRoom: boolean;
}

function buildShapes(boundary: GeoFeatureCollection): Shape[] {
  const points = boundary.features.flatMap((feature) => polygonsOf(feature.geometry).flat(2));
  const minX = Math.min(...points.map((point) => point[0]));
  const maxX = Math.max(...points.map((point) => point[0]));
  const minY = Math.min(...points.map((point) => point[1]));
  const maxY = Math.max(...points.map((point) => point[1]));

  // Odisha sits near 20°N; without the cosine term an equirectangular plot stretches
  // the state noticeably north-south and the outline stops looking like Odisha.
  const kx = Math.cos((((minY + maxY) / 2) * Math.PI) / 180);
  const spanX = (maxX - minX) * kx || 1;
  const spanY = maxY - minY || 1;
  const scale = Math.min((VIEW_WIDTH - PADDING * 2) / spanX, (VIEW_HEIGHT - PADDING * 2) / spanY);
  const offsetX = (VIEW_WIDTH - spanX * scale) / 2;
  const offsetY = (VIEW_HEIGHT - spanY * scale) / 2;
  const project = (point: Point): Point => [
    offsetX + (point[0] - minX) * kx * scale,
    VIEW_HEIGHT - offsetY - (point[1] - minY) * scale,
  ];

  return boundary.features.map((feature, index) => {
    const rings = polygonsOf(feature.geometry).flatMap((polygon) => polygon.map((ring) => ring.map(project)));
    const path = rings
      .map((ring) => `${ring.map((point, i) => `${i ? "L" : "M"}${point[0].toFixed(1)} ${point[1].toFixed(1)}`).join("")}Z`)
      .join("");
    const largest = rings.reduce<{ ring: Point[]; area: number } | null>((best, ring) => {
      const { area } = ringCentroid(ring);
      return !best || area > best.area ? { ring, area } : best;
    }, null);
    const centroid = largest ? ringCentroid(largest.ring) : { x: VIEW_WIDTH / 2, y: VIEW_HEIGHT / 2, area: 0 };
    const ring = largest?.ring ?? [];
    const width = ring.length ? Math.max(...ring.map((p) => p[0])) - Math.min(...ring.map((p) => p[0])) : 0;
    const height = ring.length ? Math.max(...ring.map((p) => p[1])) - Math.min(...ring.map((p) => p[1])) : 0;
    const name = featureName(feature.properties);
    return {
      key: `${featureId(feature.properties) ?? name}-${index}`,
      districtId: featureId(feature.properties),
      name,
      path,
      labelX: centroid.x,
      labelY: centroid.y,
      labelRoom: width > 30 && height > 22,
    };
  });
}

interface ChoroplethProps {
  boundary: GeoFeatureCollection | null;
  data: DistrictDatum[];
  bins: ScaleBin[];
  lowers: number[];
  metric: string;
  describedBy: string;
  highlighted: string | null;
  onHighlight: (districtId: string | null) => void;
  /** Drill-down. When supplied, every district becomes a keyboard-operable control. */
  onSelect?: (districtId: string | null) => void;
  selected?: string | null;
}

export function DistrictChoropleth({
  boundary, data, bins, lowers, metric, describedBy, highlighted, onHighlight, onSelect, selected = null,
}: ChoroplethProps) {
  const shapes = useMemo(() => (boundary ? buildShapes(boundary) : []), [boundary]);
  const byId = useMemo(() => new Map(data.map((entry) => [entry.districtId.toLowerCase(), entry])), [data]);
  const byName = useMemo(() => new Map(data.map((entry) => [normalizeName(entry.name), entry])), [data]);

  const lookup = (shape: { districtId: string | null; name: string }) =>
    (shape.districtId ? byId.get(shape.districtId.toLowerCase()) : undefined) ?? byName.get(normalizeName(shape.name));

  if (!boundary) {
    return (
      <div className="map-fallback">
        <p className="map-fallback__note">
          <strong>Non-geographic fallback.</strong> The licensed boundary asset did not load. Tiles keep every
          district value and its evidence state; they carry no spatial relationship.
        </p>
        <ul className="tile-grid">
          {data.map((entry) => {
            const bin = entry.value === null ? null : bins[binIndexOf(entry.value, lowers)];
            const className = `tile${entry.value === null ? " tile--nodata" : ""}${bin?.onDark ? " tile--ondark" : ""}${selected === entry.districtId ? " tile--selected" : ""}`;
            const inner = (
              <>
                <span className="tile__name">{entry.name}</span>
                <strong className="tile__value">{entry.value === null ? "no data" : formatValue(entry.value)}</strong>
              </>
            );
            return (
              <li key={entry.districtId}>
                {onSelect ? (
                  <button
                    type="button"
                    className={className}
                    style={bin ? { background: bin.color } : undefined}
                    aria-pressed={selected === entry.districtId}
                    onClick={() => onSelect(selected === entry.districtId ? null : entry.districtId)}
                  >
                    {inner}
                  </button>
                ) : (
                  <span className={className} style={bin ? { background: bin.color } : undefined}>{inner}</span>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    );
  }

  return (
    <figure className="choropleth">
      <svg
        className="choropleth__svg"
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        role="img"
        aria-label={`Odisha district map, ${metric}. All 30 districts are drawn; districts with no returned record are hatched.`}
        aria-describedby={describedBy}
      >
        <defs>
          <pattern id="hatch-nodata" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="9" height="9" fill="#FFFFFF" />
            <rect width="2.6" height="9" fill="#ABA7C6" />
          </pattern>
        </defs>
        {shapes.map((shape) => {
          const entry = lookup(shape);
          const value = entry?.value ?? null;
          const bin = value === null ? null : bins[binIndexOf(value, lowers)];
          const active = Boolean(entry && highlighted && entry.districtId === highlighted);
          const isSelected = Boolean(entry && selected && entry.districtId === selected);
          const targetId = entry?.districtId ?? null;
          return (
            <g key={shape.key}>
              <path
                d={shape.path}
                fillRule="evenodd"
                className={`district${active ? " district--active" : ""}${isSelected ? " district--selected" : ""}${onSelect ? " district--pick" : ""}`}
                fill={bin ? bin.color : "url(#hatch-nodata)"}
                role={onSelect ? "button" : undefined}
                tabIndex={onSelect ? 0 : undefined}
                aria-pressed={onSelect ? isSelected : undefined}
                aria-label={
                  onSelect
                    ? `${entry?.name ?? shape.name}: ${value === null ? "no returned record, unmeasured not zero" : `${formatValue(value)} ${metric.toLowerCase()}`}. Open district detail.`
                    : undefined
                }
                onMouseEnter={() => onHighlight(targetId)}
                onMouseLeave={() => onHighlight(null)}
                onFocus={() => onHighlight(targetId)}
                onBlur={() => onHighlight(null)}
                onClick={onSelect ? () => onSelect(isSelected ? null : targetId) : undefined}
                onKeyDown={
                  onSelect
                    ? (event) => {
                      if (event.key !== "Enter" && event.key !== " ") return;
                      event.preventDefault();
                      onSelect(isSelected ? null : targetId);
                    }
                    : undefined
                }
              >
                <title>
                  {`${entry?.name ?? shape.name}: ${value === null ? "no returned record — unmeasured, not zero" : `${formatValue(value)} ${metric.toLowerCase()}`}`}
                </title>
              </path>
              {value !== null && shape.labelRoom && (
                <text
                  className={`district__value${bin?.onDark ? " district__value--light" : ""}`}
                  x={shape.labelX}
                  y={shape.labelY}
                  textAnchor="middle"
                  dominantBaseline="central"
                  aria-hidden="true"
                >
                  {formatValue(value)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption className="choropleth__caption">
        DataMeet Census 2011 district geometry · CC BY 2.5 India · every one of the 30 districts is drawn, painted
        only where a record was returned
      </figcaption>
    </figure>
  );
}

interface LegendProps {
  bins: ScaleBin[];
  metric: string;
  noDataCount: number;
  totalCount: number;
}

export function MapLegend({ bins, metric, noDataCount, totalCount }: LegendProps) {
  return (
    <div className="legend">
      <div className="legend__head">
        <span className="legend__eyebrow">Colour encodes</span>
        <strong className="legend__metric">{metric}</strong>
      </div>
      <ul className="legend__scale">
        <li className="legend__item legend__item--nodata">
          <span className="legend__swatch legend__swatch--hatch" aria-hidden="true" />
          <span className="legend__label">
            <b>No data</b>
            <small>{noDataCount} of {totalCount} districts · unmeasured, not zero</small>
          </span>
        </li>
        {bins.map((bin) => (
          <li className="legend__item" key={bin.index}>
            <span
              className={`legend__swatch${bin.onDark ? " legend__swatch--ondark" : ""}`}
              style={{ background: bin.color }}
              aria-hidden="true"
            >
              {bin.index + 1}
            </span>
            <span className="legend__label">
              <b>{bin.label}</b>
              <small>band {bin.index + 1} of {bins.length}</small>
            </span>
          </li>
        ))}
        {!bins.length && (
          <li className="legend__item legend__item--empty">
            <span className="legend__label">
              <b>No band is drawn</b>
              <small>this layer returned no district value at all</small>
            </span>
          </li>
        )}
      </ul>
    </div>
  );
}

interface TableProps {
  id: string;
  data: DistrictDatum[];
  bins: ScaleBin[];
  lowers: number[];
  metric: string;
  unit: string;
  hidden: boolean;
  highlighted: string | null;
  onHighlight: (districtId: string | null) => void;
  onSelect?: (districtId: string | null) => void;
  selected?: string | null;
}

export function DistrictTable({
  id, data, bins, lowers, metric, unit, hidden, highlighted, onHighlight, onSelect, selected = null,
}: TableProps) {
  const rows = useMemo(
    () => [...data].sort((left, right) => (right.value ?? -1) - (left.value ?? -1) || left.name.localeCompare(right.name)),
    [data],
  );
  return (
    <div className={hidden ? "sr-only" : "data-table-wrap"} id={id}>
      <table className="data-table">
        <caption>
          {metric} by district. {data.length} districts listed; a district with no returned record is marked
          &ldquo;no data&rdquo; and is never counted as zero.
        </caption>
        <thead>
          <tr>
            <th scope="col">District</th>
            <th scope="col">{unit}</th>
            <th scope="col">Band</th>
            <th scope="col">Evidence state</th>
            {onSelect && <th scope="col">Detail</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const bin = row.value === null ? null : bins[binIndexOf(row.value, lowers)];
            return (
              <tr
                key={row.districtId}
                className={`${highlighted === row.districtId ? "is-highlighted" : ""}${selected === row.districtId ? " is-selected" : ""}`.trim() || undefined}
                onMouseEnter={() => onHighlight(row.districtId)}
                onMouseLeave={() => onHighlight(null)}
              >
                <th scope="row">{row.name}</th>
                <td className="data-table__value">{row.value === null ? "—" : formatValue(row.value)}</td>
                <td>
                  {bin ? (
                    <span className={`band-chip${bin.onDark ? " band-chip--ondark" : ""}`} style={{ background: bin.color }}>
                      {bin.index + 1}
                    </span>
                  ) : (
                    <span className="band-chip band-chip--hatch">–</span>
                  )}
                </td>
                <td>{row.value === null ? "No returned record: unmeasured, not zero" : "Record returned"}</td>
                {onSelect && (
                  <td>
                    <button
                      type="button"
                      className="rowpick"
                      aria-pressed={selected === row.districtId}
                      onClick={() => onSelect(selected === row.districtId ? null : row.districtId)}
                    >
                      {selected === row.districtId ? "Close" : "Open"}
                    </button>
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
