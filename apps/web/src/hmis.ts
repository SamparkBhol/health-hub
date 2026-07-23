/**
 * The district-monthly HMIS panel: metric vocabulary, period axis and the one
 * sentence that has to travel with every value it produces.
 *
 * HMIS rows are facility service and test *records* submitted month by month.
 * One person tested twice is two records; a person never brought to a reporting
 * facility is no record at all. They are provisional — later months revise
 * earlier ones — and they have no population denominator attached here. So they
 * are not deduplicated people and they are not population incidence, and the
 * caption below is not optional decoration.
 *
 * Deliberately dependency-free so it can be unit-tested on its own.
 */

export interface HmisMetricMeta {
  value: string;
  label: string;
  /** Short column unit for the accessible table header. */
  unit: string;
  /** What one unit of this metric counts. */
  definition: string;
  /**
   * Factor the raw API value is multiplied by before it is drawn.
   *
   * The positivity ratios run from 0 to about 0.068, and the map's shared
   * two-decimal formatter prints every one of them as "0" — a district with real
   * positives would be labelled zero on the map while still being shaded. They
   * are therefore drawn per 1,000 tests, which is what `unit` says.
   */
  scale: number;
}

export const HMIS_METRICS: HmisMetricMeta[] = [
  {
    value: "malaria_test_positivity",
    label: "Malaria test positivity",
    unit: "positive per 1,000 tests",
    definition: "Malaria-positive records per 1,000 malaria tests reported by facilities that month.",
    scale: 1000,
  },
  {
    value: "malaria_microscopy_positivity",
    label: "Malaria microscopy positivity",
    unit: "positive per 1,000 slides",
    definition: "Microscopy-positive records per 1,000 microscopy slides examined that month.",
    scale: 1000,
  },
  {
    value: "malaria_positive_records",
    label: "Malaria positive records",
    unit: "records",
    definition: "Facility-reported malaria-positive test records that month, by any test method.",
    scale: 1,
  },
  {
    value: "malaria_microscopy_positive_records",
    label: "Malaria microscopy positive records",
    unit: "records",
    definition: "Facility-reported malaria-positive microscopy records that month.",
    scale: 1,
  },
  {
    value: "dengue_positive_records",
    label: "Dengue positive records",
    unit: "records",
    definition: "Facility-reported dengue-positive test records that month.",
    scale: 1,
  },
  {
    value: "childhood_diarrhoea_records",
    label: "Childhood diarrhoea records",
    unit: "records",
    definition: "Facility-reported childhood diarrhoea service records that month.",
    scale: 1,
  },
];

export function hmisMetricMeta(metric: string): HmisMetricMeta {
  const known = HMIS_METRICS.find((entry) => entry.value === metric);
  if (known) return known;
  return {
    value: metric,
    label: metric.replaceAll("_", " "),
    unit: metric.replaceAll("_", " "),
    definition: "This build holds no definition for this HMIS column; the API's own scope note applies.",
    // An unknown column is drawn exactly as the API sent it.
    scale: 1,
  };
}

/** Applies the display scale. A missing value stays missing — never becomes zero. */
export function scaleHmisValue(value: number | null | undefined, meta: HmisMetricMeta): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (meta.scale === 1) return value;
  return Number((value * meta.scale).toFixed(3));
}

/** The sentence that has to accompany every HMIS value on screen. */
export const HMIS_EPISTEMIC_CAPTION =
  "These are provisional facility-reported test and service records — not deduplicated people and "
  + "not population incidence. A district that reports from more facilities produces more records "
  + "without being sicker, and a district with no record is unmeasured, never disease-free.";

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

/** `2020-03-01` → `March 2020`. An unparseable value is returned untouched. */
export function formatHmisPeriod(period: string): string {
  const match = /^(\d{4})-(\d{2})-\d{2}$/.exec(period.trim());
  if (!match) return period;
  const month = MONTHS[Number(match[2]) - 1];
  return month ? `${month} ${match[1]}` : period;
}

/**
 * Every month-start between the two bounds the API reports, newest first.
 *
 * The panel publishes only its first and last period, so the selector is
 * reconstructed from that closed interval. Bad or inverted bounds yield whatever
 * single period is usable rather than an invented range.
 */
export function hmisPeriods(start: string | null | undefined, end: string | null | undefined): string[] {
  const first = parseMonth(start);
  const last = parseMonth(end);
  if (!first && !last) return [];
  if (!first || !last) return [format(first ?? last!)];
  if (first.year * 12 + first.month > last.year * 12 + last.month) return [format(last)];

  const periods: string[] = [];
  let year = last.year;
  let month = last.month;
  // Guard against a pathological range: 100 years of months is already absurd.
  while (periods.length < 1200 && (year * 12 + month) >= (first.year * 12 + first.month)) {
    periods.push(format({ year, month }));
    month -= 1;
    if (month < 1) { month = 12; year -= 1; }
  }
  return periods;
}

interface Month { year: number; month: number }

function parseMonth(value: string | null | undefined): Month | null {
  const match = /^(\d{4})-(\d{2})-\d{2}$/.exec((value ?? "").trim());
  if (!match) return null;
  const month = Number(match[2]);
  if (month < 1 || month > 12) return null;
  return { year: Number(match[1]), month };
}

function format({ year, month }: Month): string {
  return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-01`;
}
