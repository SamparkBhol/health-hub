/**
 * What each official public-health metric actually measures, and therefore what
 * the surrounding interface is allowed to call it.
 *
 * The NCVBDC annual table publishes six columns and only three of them describe
 * disease burden. API, reported cases and deaths are counts of illness. ABER is
 * how many people were tested — pure surveillance effort — and SPR and Pf% are
 * properties of the tested sample and the parasite mix. Painting those three
 * under a "disease burden" heading tells an operator that a well-tested district
 * is a sick district, which is the exact inversion this interface exists to
 * prevent. The framing is therefore derived from the metric, never fixed.
 *
 * Deliberately dependency-free so it can be unit-tested on its own.
 */

export type MetricFraming = "burden" | "effort_and_mix";

export interface MetricMeta {
  /** Query value sent to the API. */
  value: string;
  /** Readable name for a selector option and an accessible table header. */
  label: string;
  /** Short column unit, for the table header and the readout. */
  unit: string;
  framing: MetricFraming;
  /** The eyebrow line above the heading. */
  eyebrow: string;
  /** One sentence stating what the number does and does not establish. */
  caption: string;
}

const BURDEN_EYEBROW = "Observed disease burden · official public data";
const EFFORT_EYEBROW = "Surveillance effort and parasite mix — not burden";

const BURDEN_CAPTION =
  "This metric counts reported illness. It still carries detection and reporting effort: a district "
  + "that tests less reports less.";
const EFFORT_CAPTION =
  "This metric describes testing effort and the parasite mix in the samples examined, not how much "
  + "disease a district has. A high value here is a statement about surveillance, not about sickness.";

function meta(
  value: string, label: string, unit: string, framing: MetricFraming,
): MetricMeta {
  return {
    value,
    label,
    unit,
    framing,
    eyebrow: framing === "burden" ? BURDEN_EYEBROW : EFFORT_EYEBROW,
    caption: framing === "burden" ? BURDEN_CAPTION : EFFORT_CAPTION,
  };
}

/** The six NCVBDC annual district columns the map can paint, in selector order. */
export const MALARIA_METRICS: MetricMeta[] = [
  meta("api", "Annual Parasite Incidence", "API per 1,000", "burden"),
  meta("total_cases", "Reported annual cases / positives", "cases / positives", "burden"),
  meta("deaths", "Reported deaths", "deaths", "burden"),
  meta("spr", "Slide Positivity Rate", "SPR %", "effort_and_mix"),
  meta("aber", "Annual Blood Examination Rate", "ABER %", "effort_and_mix"),
  meta("pf_percent", "P. falciparum percentage", "Pf %", "effort_and_mix"),
];

/**
 * Never guesses a framing. A metric this build does not know about is described
 * as unclassified rather than folded into "disease burden".
 */
export function malariaMetricMeta(metric: string): MetricMeta {
  const known = MALARIA_METRICS.find((entry) => entry.value === metric);
  if (known) return known;
  return {
    value: metric,
    label: metric.replaceAll("_", " "),
    unit: metric.replaceAll("_", " "),
    framing: "effort_and_mix",
    eyebrow: "Metric not classified by this interface · official public data",
    caption:
      "This build does not hold a definition for this column, so it is not described as disease "
      + "burden. The API's own metric definition is shown above.",
  };
}
