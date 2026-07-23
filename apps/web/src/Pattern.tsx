import { useMemo } from "react";
import { CalendarClock, ExternalLink, X } from "lucide-react";
import { RAMP, binIndexOf, buildScale } from "./DistrictMap";
import { resolveDistrictId, tallyDistrictDisease } from "./districts";
import { humanizeDisease } from "./format";
import { LANGUAGES, isAnswerLanguage, langAttribute } from "./languages";
import { TranslateEvidence } from "./Translate";
import { Chip, SyntheticBadge } from "./ui";
import { isSynthetic } from "./epistemics";
import type { Signal } from "./types";

export interface PatternDistrict {
  id: string;
  name: string;
}

function sourceHost(value: string): string {
  try {
    return new URL(value).hostname.replace(/^www\./, "");
  } catch {
    return value;
  }
}

/** Resolves a signal's free-text district name onto the canonical district id. */
export function districtIdFor(name: string | null | undefined, districts: PatternDistrict[]): string | null {
  return resolveDistrictId(name, districts);
}

function utcDay(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "unknown" : date.toISOString().slice(0, 10);
}

function shortDay(day: string): string {
  const date = new Date(`${day}T00:00:00Z`);
  return Number.isNaN(date.getTime())
    ? day
    : new Intl.DateTimeFormat("en-IN", { day: "2-digit", month: "short", timeZone: "UTC" }).format(date);
}

/* ----------------------------------------------------------- disease filters */

/**
 * Disease tags that actually appear in the retrieved set, with their counts.
 * Tags with no retrieved record are not offered, because offering a filter that
 * can only ever return nothing reads as a data gap in the disease rather than in
 * the collection.
 */
export function DiseaseChips({
  signals, value, onChange,
}: {
  signals: Signal[];
  value: string;
  onChange: (next: string) => void;
}) {
  const tallies = useMemo(() => {
    const counts = new Map<string, number>();
    for (const signal of signals) {
      const key = signal.disease?.trim().toLowerCase().replaceAll(" ", "_");
      if (!key) continue;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]));
  }, [signals]);

  return (
    <div className="dchips" role="group" aria-label="Filter the map by disease tag">
      <button
        type="button"
        className={`dchip${value === "" ? " is-on" : ""}`}
        aria-pressed={value === ""}
        onClick={() => onChange("")}
      >
        <b>All tags</b>
        <span>{signals.length}</span>
      </button>
      {tallies.map(([disease, count]) => (
        <button
          key={disease}
          type="button"
          className={`dchip${value === disease ? " is-on" : ""}`}
          aria-pressed={value === disease}
          onClick={() => onChange(value === disease ? "" : disease)}
        >
          <b>{humanizeDisease(disease) ?? disease}</b>
          <span>{count}</span>
        </button>
      ))}
      {!tallies.length && <p className="dchips__empty">No disease tag has a retrieved record yet.</p>}
    </div>
  );
}

/* --------------------------------------------------------- retrieval timeline */

export interface TimelineBucket {
  day: string;
  added: number;
  cumulative: number;
}

export function buildTimeline(signals: Signal[]): TimelineBucket[] {
  const perDay = new Map<string, number>();
  for (const signal of signals) {
    const day = utcDay(signal.retrievedAt);
    if (day === "unknown") continue;
    perDay.set(day, (perDay.get(day) ?? 0) + 1);
  }
  let running = 0;
  return [...perDay.entries()]
    .sort((left, right) => left[0].localeCompare(right[0]))
    .map(([day, added]) => {
      running += added;
      return { day, added, cumulative: running };
    });
}

/**
 * How the evidence base accumulated across the retrieval window.
 *
 * Selecting a day re-asks the API with `retrieved_to` set to the end of that day,
 * so the choropleth redraws as the state of knowledge on that date. The axis is
 * retrieval time — when this platform fetched a document — and never symptom
 * onset or event date.
 */
export function RetrievalTimeline({
  buckets, cutoff, onCutoff,
}: {
  buckets: TimelineBucket[];
  cutoff: string | null;
  onCutoff: (day: string | null) => void;
}) {
  const peak = buckets.length ? buckets[buckets.length - 1].cumulative : 0;
  return (
    <div className="timeline">
      <div className="timeline__head">
        <span className="eyebrow"><CalendarClock size={13} strokeWidth={3} aria-hidden="true" /> Retrieval accumulation</span>
        <button
          type="button"
          className={`timeline__all${cutoff === null ? " is-on" : ""}`}
          aria-pressed={cutoff === null}
          onClick={() => onCutoff(null)}
        >
          Full window · {peak}
        </button>
      </div>

      {buckets.length === 0 ? (
        <p className="timeline__empty">
          No retrieval receipt carries a usable timestamp yet, so there is no time axis to draw.
        </p>
      ) : (
        <>
          <ol className="timeline__track">
            {buckets.map((bucket) => {
              const active = cutoff === bucket.day;
              const height = peak ? Math.max(8, Math.round((bucket.cumulative / peak) * 100)) : 8;
              return (
                <li key={bucket.day}>
                  <button
                    type="button"
                    className={`timeline__bar${active ? " is-on" : ""}${cutoff && bucket.day > cutoff ? " is-after" : ""}`}
                    aria-pressed={active}
                    onClick={() => onCutoff(active ? null : bucket.day)}
                    title={`As of ${bucket.day}: ${bucket.cumulative} retrieved records (+${bucket.added} that day)`}
                  >
                    <span className="timeline__fill" style={{ height: `${height}%` }} aria-hidden="true" />
                    <span className="timeline__count">{bucket.cumulative}</span>
                    <span className="timeline__day">{shortDay(bucket.day)}</span>
                  </button>
                </li>
              );
            })}
          </ol>
          <p className="timeline__caption">
            {buckets.length === 1
              ? "One retrieval day in the retained window. The time axis fills out as collection runs on more days; a single day is not a trend."
              : `${buckets.length} retrieval days · click a day to redraw the map as it stood then.`}{" "}
            Axis is retrieval time, not symptom onset or event date.
          </p>
        </>
      )}
    </div>
  );
}

/* ----------------------------------------------------------- disease pattern */

/**
 * The district x disease grid — the pattern the map cannot show, because a
 * choropleth can only carry one variable at a time. Cells are counts of
 * retrieved documents. An empty cell is hatched, never zero-filled, because no
 * retrieved document for a pair is a statement about publishing, not disease.
 *
 * Only records whose place resolved onto a canonical district become rows. A
 * record tagged "District Unavailable" is not a thirty-first district and must
 * not sit in the row order as though it were one; those records are counted
 * once, below the grid, on a line of their own.
 */
export function DiseaseMatrix({
  signals, districts,
}: {
  signals: Signal[];
  districts: PatternDistrict[];
}) {
  const { rows, columns, cells, bins, lowers, silentDistricts, unresolved } = useMemo(() => {
    const tally = tallyDistrictDisease(signals, districts);
    const scale = buildScale([...tally.cells.values()]);
    return {
      rows: tally.rows.map((row) => ({
        id: row.districtId,
        name: districts.find((district) => district.id === row.districtId)?.name
          ?? row.districtId.replace(/^OD-DIST-/i, ""),
        total: row.total,
      })),
      columns: tally.columns.map((column) => ({
        id: column.disease,
        label: humanizeDisease(column.disease) ?? column.disease,
        total: column.total,
      })),
      cells: tally.cells,
      bins: scale.bins,
      lowers: scale.lowers,
      silentDistricts: districts.length - tally.rows.length,
      unresolved: tally.unresolved,
    };
  }, [districts, signals]);

  const unresolvedLine = unresolved.records > 0 && (
    <p className="matrix__unresolved">
      <strong>{unresolved.records} retrieved {unresolved.records === 1 ? "record" : "records"} carry no district.</strong>{" "}
      Their place did not resolve to one of Odisha&rsquo;s {districts.length} districts
      {unresolved.names.length > 0 && (
        <> (published as {unresolved.names.slice(0, 3).map((entry) => `“${entry.name}” ×${entry.count}`).join(", ")})</>
      )}
      , so they are counted here instead of being drawn as a district row. They are evidence about a
      disease tag, not about any district
      {unresolved.diseases.length > 0 && (
        <>: {unresolved.diseases.slice(0, 4).map((entry) => `${humanizeDisease(entry.name) ?? entry.name} ×${entry.count}`).join(", ")}</>
      )}
      .
    </p>
  );

  if (!rows.length || !columns.length) {
    return (
      <>
        <p className="matrix__empty">
          No district and disease pair has a retrieved record yet, so there is no pattern to draw.
        </p>
        {unresolvedLine}
      </>
    );
  }

  return (
    <div className="matrix">
      <div className="matrix__scroll">
        <table className="matrix__table">
          <caption>
            Retrieved published documents by district and disease tag. A hatched cell means nothing has been retrieved
            for that pair; it is not a zero and it is not an absence of disease.
          </caption>
          <thead>
            <tr>
              <th scope="col">District</th>
              {columns.map((column) => (
                <th key={column.id} scope="col"><span>{column.label}</span><small>{column.total}</small></th>
              ))}
              <th scope="col" className="matrix__total">All tags</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <th scope="row">{row.name}</th>
                {columns.map((column) => {
                  const count = cells.get(`${row.id}|${column.id}`) ?? null;
                  const bin = count === null ? null : bins[binIndexOf(count, lowers)];
                  return (
                    <td
                      key={column.id}
                      className={`matrix__cell${count === null ? " matrix__cell--none" : ""}${bin?.onDark ? " matrix__cell--ondark" : ""}`}
                      style={bin ? { background: bin.color } : undefined}
                    >
                      {count === null ? <span className="sr-only">no retrieved record</span> : count}
                    </td>
                  );
                })}
                <td className="matrix__total">{row.total}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="matrix__note">
        {silentDistricts > 0
          ? `${silentDistricts} of ${districts.length} districts have no retrieved record on any tag and are not listed above. They are unmeasured, not disease-free.`
          : "Every district in the boundary set has at least one retrieved record."}
      </p>
      {unresolvedLine}
    </div>
  );
}

/* ------------------------------------------------------------ district drill */

export function DistrictDetail({
  districtId, districtName, signals, firstRetrievedAt, lastRetrievedAt, onClear,
}: {
  districtId: string;
  districtName: string;
  signals: Signal[];
  firstRetrievedAt?: string | null;
  lastRetrievedAt?: string | null;
  onClear: () => void;
}) {
  const diseases = useMemo(() => {
    const counts = new Map<string, number>();
    for (const signal of signals) {
      const key = signal.disease?.trim().toLowerCase().replaceAll(" ", "_") ?? "untagged";
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts.entries()].sort((left, right) => right[1] - left[1]);
  }, [signals]);

  const languages = useMemo(() => {
    const counts = new Map<string, number>();
    for (const signal of signals) counts.set(signal.language, (counts.get(signal.language) ?? 0) + 1);
    return [...counts.entries()].sort((left, right) => right[1] - left[1]);
  }, [signals]);

  const peak = diseases.length ? diseases[0][1] : 1;

  return (
    <div className="drill">
      <div className="drill__head">
        <div>
          <span className="eyebrow">District detail</span>
          <h3>{districtName}</h3>
          <code>{districtId}</code>
        </div>
        <button type="button" className="drill__close" onClick={onClear} aria-label="Close district detail">
          <X size={16} strokeWidth={3} aria-hidden="true" />
        </button>
      </div>

      {signals.length === 0 ? (
        <p className="drill__empty">
          No published record has been retrieved for this district. That is a statement about what the registered
          sources published and this platform fetched — not about whether disease is present.
        </p>
      ) : (
        <>
          <dl className="kv">
            <div><dt>Retrieved records</dt><dd>{signals.length}</dd></div>
            <div><dt>Disease tags</dt><dd>{diseases.length}</dd></div>
            <div><dt>First receipt</dt><dd>{firstRetrievedAt?.slice(0, 10) ?? "not reported"}</dd></div>
            <div><dt>Latest receipt</dt><dd>{lastRetrievedAt?.slice(0, 10) ?? "not reported"}</dd></div>
          </dl>

          <div className="drill__bars">
            <span className="eyebrow">By disease tag</span>
            <ul>
              {diseases.map(([disease, count], index) => (
                <li key={disease}>
                  <span>{humanizeDisease(disease) ?? disease}</span>
                  <span className="drill__track">
                    <span
                      className="drill__fill"
                      style={{ width: `${Math.max(6, (count / peak) * 100)}%`, background: RAMP[Math.min(index, RAMP.length - 1)] }}
                    />
                  </span>
                  <b>{count}</b>
                </li>
              ))}
            </ul>
          </div>

          <div className="drill__langs">
            <span className="eyebrow">Source language</span>
            <div>
              {languages.map(([code, count]) => (
                <span key={code} className="drill__lang" lang={langAttribute(code)}>
                  {isAnswerLanguage(code) ? `${LANGUAGES[code].mark} ${LANGUAGES[code].endonym}` : code}
                  <b>{count}</b>
                </span>
              ))}
            </div>
          </div>

          <div className="drill__records">
            {signals.map((signal) => (
              <article key={signal.id} className="drill__record">
                <div className="drill__rectop">
                  <Chip tone={signal.assertion === "affirmed" ? "ok" : "warn"} size="sm">
                    {signal.assertion.replaceAll("_", " ")}
                  </Chip>
                  <span>{humanizeDisease(signal.disease) ?? "untagged"}</span>
                  {isSynthetic(signal) && <SyntheticBadge />}
                </div>
                <blockquote lang={langAttribute(signal.language)}>{signal.evidence}</blockquote>
                <p className="drill__recsource">
                  {signal.canonicalUrl ? (
                    <a href={signal.canonicalUrl} target="_blank" rel="noreferrer">
                      {signal.source} · {sourceHost(signal.canonicalUrl)} <ExternalLink size={11} strokeWidth={2.5} aria-hidden="true" />
                    </a>
                  ) : signal.source}
                </p>
                <TranslateEvidence text={signal.evidence} sourceLanguage={signal.language} id={`drill-${signal.id}`} />
              </article>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
