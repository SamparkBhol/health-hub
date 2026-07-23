/**
 * District identity — the single place that decides whether a free-text place
 * name taken from a source document resolves onto one of Odisha's 30 canonical
 * districts.
 *
 * The distinction matters for honesty, not tidiness. A record whose place could
 * not be resolved is evidence about *something*, but it is not evidence about a
 * district, so it must never be rendered as if it were a district row. This
 * module keeps the resolution rule and the resolved/unresolved split together so
 * both sides stay visible to whoever renders them.
 *
 * Deliberately dependency-free so it can be unit-tested on its own.
 */

export interface NamedDistrict {
  id: string;
  name: string;
}

/**
 * Spellings the census boundary file and the published sources disagree on.
 * Every value on the right is the spelling used by the bundled boundary set.
 */
const NAME_ALIASES: Record<string, string> = {
  anugul: "angul",
  baleshwar: "balasore",
  bauda: "boudh",
  debagarh: "deogarh",
  jajapur: "jajpur",
  kendujhar: "keonjhar",
  khurda: "khordha",
  nabarangapur: "nabarangpur",
  subarnapur: "subarnapur",
  sonepur: "subarnapur",
  jagatsinghapur: "jagatsinghpur",
};

export function normalizeName(value: string): string {
  const key = value.trim().toLowerCase();
  return NAME_ALIASES[key] ?? key;
}

/** The canonical district id for a free-text name, or `null` when it resolves to none. */
export function resolveDistrictId(
  name: string | null | undefined,
  districts: NamedDistrict[],
): string | null {
  if (!name || !name.trim()) return null;
  const key = normalizeName(name);
  return districts.find((district) => normalizeName(district.name) === key)?.id ?? null;
}

export interface CountedName {
  name: string;
  count: number;
}

export interface DistrictDiseaseTally {
  /** District rows, busiest first. Every id here is a canonical district id. */
  rows: Array<{ districtId: string; total: number }>;
  /** Disease columns, busiest first, counting resolved rows only. */
  columns: Array<{ disease: string; total: number }>;
  /** `${districtId}|${disease}` → retrieved-document count. */
  cells: Map<string, number>;
  /**
   * Records whose place never resolved to a district. Reported as its own total
   * rather than smuggled into the grid as a pseudo-district row.
   */
  unresolved: {
    records: number;
    names: CountedName[];
    diseases: CountedName[];
  };
}

function diseaseKey(value: string | null | undefined): string | null {
  const key = value?.trim().toLowerCase().replaceAll(" ", "_");
  return key ? key : null;
}

function ranked(counts: Map<string, number>): CountedName[] {
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([name, count]) => ({ name, count }));
}

/**
 * Cross-tabulates retrieved documents by district and disease tag.
 *
 * Rows are built only from records whose district resolved; everything else is
 * accumulated into `unresolved` so the caller can state that total explicitly.
 * Column totals therefore always equal the sum of the visible cells.
 */
export function tallyDistrictDisease(
  signals: Array<{ district?: string | null; disease?: string | null }>,
  districts: NamedDistrict[],
): DistrictDiseaseTally {
  const cells = new Map<string, number>();
  const rowTotals = new Map<string, number>();
  const columnTotals = new Map<string, number>();
  const unresolvedNames = new Map<string, number>();
  const unresolvedDiseases = new Map<string, number>();
  let unresolvedRecords = 0;

  for (const signal of signals) {
    const disease = diseaseKey(signal.disease);
    if (!disease) continue;
    const districtId = resolveDistrictId(signal.district, districts);
    if (!districtId) {
      unresolvedRecords += 1;
      const label = signal.district?.trim() || "place not stated";
      unresolvedNames.set(label, (unresolvedNames.get(label) ?? 0) + 1);
      unresolvedDiseases.set(disease, (unresolvedDiseases.get(disease) ?? 0) + 1);
      continue;
    }
    const cell = `${districtId}|${disease}`;
    cells.set(cell, (cells.get(cell) ?? 0) + 1);
    rowTotals.set(districtId, (rowTotals.get(districtId) ?? 0) + 1);
    columnTotals.set(disease, (columnTotals.get(disease) ?? 0) + 1);
  }

  return {
    rows: ranked(rowTotals).map((entry) => ({ districtId: entry.name, total: entry.count })),
    columns: ranked(columnTotals).map((entry) => ({ disease: entry.name, total: entry.count })),
    cells,
    unresolved: {
      records: unresolvedRecords,
      names: ranked(unresolvedNames),
      diseases: ranked(unresolvedDiseases),
    },
  };
}
