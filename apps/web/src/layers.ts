import type { LayerType } from "./types";

/**
 * The five layers are never blended. Each one carries its own metric, its own legend
 * title, and its own pair of claims: what the numbers are, and what they are not.
 * The "isNot" line is rendered at the same size as the "is" line, deliberately.
 */
export interface LayerSpec {
  id: LayerType;
  /** Tab label. Two words maximum. */
  tab: string;
  /** Sentence-case name used in prose. */
  name: string;
  /** What the map colour encodes, in the operator's words. Shown in the legend head. */
  metric: string;
  /** Unit noun for the accessible table column header. */
  unit: string;
  is: string;
  isNot: string;
  /** Headline when the layer returns no rows at all. */
  emptyTitle: string;
  emptyBody: string;
  /** True when this layer can legitimately be drawn on district geometry. */
  mappable: boolean;
}

export const LAYER_ORDER: LayerType[] = [
  "public_source_signal",
  "verified_event",
  "official_event_catalogue",
  "observed_surveillance",
  "coverage",
];

export const LAYERS: Record<LayerType, LayerSpec> = {
  public_source_signal: {
    id: "public_source_signal",
    tab: "Published signals",
    name: "published source signals",
    metric: "Published evidence counts",
    unit: "published items retrieved",
    is: "A count of published items retrieved from registered sources and tagged to a district, indexed by the time this system fetched them.",
    isNot: "Not cases. Not incidence, prevalence, burden, attack rate, or unique outbreaks. A louder district is a better-covered district.",
    emptyTitle: "No published items were returned for these filters",
    emptyBody: "Widen the retrieval window or clear the disease filter. An empty result describes this collection, not Odisha.",
    mappable: true,
  },
  verified_event: {
    id: "verified_event",
    tab: "Verified events",
    name: "reviewer-verified events",
    metric: "Reviewer-accepted record counts",
    unit: "accepted records",
    is: "Candidate evidence a qualified reviewer read and promoted to an event record.",
    isNot: "Not official surveillance, and not a case count. An empty layer means nothing was promoted, not that nothing happened.",
    emptyTitle: "No reviewer-verified events were returned",
    emptyBody: "Nothing has been promoted through human review in this deployment. The map stays entirely unmeasured rather than falling back to candidate signals.",
    mappable: true,
  },
  official_event_catalogue: {
    id: "official_event_catalogue",
    tab: "Official catalogue",
    name: "the public outbreak catalogue",
    metric: "Catalogue entry counts",
    unit: "catalogue entries",
    is: "Entries copied from a positive-only public outbreak catalogue, retaining the publisher's own outbreak identifier and reporting week.",
    isNot: "Not a district-week panel. There are no negative weeks in this source, so a district with no entry is unmeasured, never zero.",
    emptyTitle: "No public catalogue records were returned",
    emptyBody: "A positive-only catalogue has no valid negative rows. Missing records are not zero outbreaks.",
    mappable: true,
  },
  observed_surveillance: {
    id: "observed_surveillance",
    tab: "Observed surveillance",
    name: "observed surveillance",
    metric: "Official weekly cases per 100,000",
    unit: "cases per 100,000",
    is: "Authorised district-week surveillance rates, with counts, denominators, case-definition version and reporting completeness returned beside each value.",
    isNot: "Not derived from anything else on this site. No media count is ever promoted into this layer, and a missing district-week is not rendered as zero incidence.",
    emptyTitle: "No authorised observed aggregates were returned",
    emptyBody: "Routine surveillance aggregates have not been supplied to this deployment. Every district stays unmeasured until an authorised no-PII export passes the contract.",
    mappable: true,
  },
  coverage: {
    id: "coverage",
    tab: "Source coverage",
    name: "source coverage",
    metric: "Not mappable to districts",
    unit: "collection receipts",
    is: "The registered acquisition surface: which routes exist, which are enabled, and when each last returned a successful receipt.",
    isNot: "Not district-level. A source is statewide or national, so its status cannot be painted onto any district polygon.",
    emptyTitle: "No source-coverage records were returned",
    emptyBody: "District coverage cannot be inferred from an empty source registry.",
    mappable: false,
  },
  forecast: {
    id: "forecast",
    tab: "Forecast",
    name: "forecast",
    metric: "Withheld",
    unit: "probability",
    is: "An authorised district-week outbreak probability with a scored, calibrated model behind it.",
    isNot: "Not available for Odisha. The route refuses rather than returning a number the data cannot support.",
    emptyTitle: "No forecast layer is served",
    emptyBody: "The real forecast route returns a typed refusal. See the forecast boundary page.",
    mappable: false,
  },
};
