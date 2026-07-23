/**
 * Starter prompts for the evidence agent, chosen against the evidence that has
 * actually been retained.
 *
 * A worked example that returns nothing is worse than no example at all: the
 * operator reads the empty refusal as "this system does not work" rather than
 * as "nothing has been published about that district and disease". Every prompt
 * offered here therefore declares the district and disease tag it depends on,
 * and is only shown when a retained record carries that pair. Prompts that need
 * no evidence — the frozen data-audit question — carry `needs: null` and are
 * always offered.
 *
 * Deliberately free of runtime imports so it can be unit-tested on its own.
 */

import type { AnswerLanguage } from "./languages";

/** One prompt is offered per slot, so the four tiles stay stable as data change. */
export type ExampleSlot = "odia" | "hindi" | "english" | "audit";

export interface EvidenceCondition {
  /** Canonical district name as the signal payload spells it. */
  district?: string;
  /** Disease tag, lower_snake_case. */
  disease?: string;
}

export interface ExamplePrompt {
  slot: ExampleSlot;
  code: AnswerLanguage;
  text: string;
  gloss: string;
  needs: EvidenceCondition | null;
}

export const SLOT_ORDER: ExampleSlot[] = ["odia", "hindi", "english", "audit"];

/**
 * Candidates in preference order within each slot. The first one whose evidence
 * condition holds wins its slot; if none holds, that slot is left empty rather
 * than filled with a prompt that is known to return nothing.
 */
export const EXAMPLE_CANDIDATES: ExamplePrompt[] = [
  {
    slot: "odia",
    code: "or",
    text: "ଖୋର୍ଦ୍ଧାରେ ଡେଙ୍ଗୁ ବିଷୟରେ କେଉଁ ପ୍ରମାଣ ଅଛି?",
    gloss: "What dengue evidence exists in Khordha?",
    needs: { district: "Khordha", disease: "dengue" },
  },
  {
    slot: "odia",
    code: "or",
    text: "ପୁରୀରେ କୋଭିଡ୍-୧୯ ବିଷୟରେ କେଉଁ ପ୍ରମାଣ ପ୍ରକାଶିତ ହୋଇଛି?",
    gloss: "What COVID-19 evidence has been published for Puri?",
    needs: { district: "Puri", disease: "covid_19" },
  },
  {
    slot: "hindi",
    code: "hi",
    text: "पुरी जिले में कोविड-19 के बारे में क्या प्रकाशित हुआ है?",
    gloss: "What has been published about COVID-19 in Puri district?",
    needs: { district: "Puri", disease: "covid_19" },
  },
  {
    slot: "hindi",
    code: "hi",
    text: "खुर्दा जिले में कैंसर के बारे में कौन से रिकॉर्ड प्रकाशित हुए हैं?",
    gloss: "Which cancer records have been published for Khordha district?",
    needs: { district: "Khordha", disease: "cancer" },
  },
  {
    slot: "english",
    code: "en",
    text: "Which districts have published dengue evidence, and from which sources?",
    gloss: "District and source breakdown",
    needs: { disease: "dengue" },
  },
  {
    slot: "english",
    code: "en",
    text: "Which districts have published malaria evidence, and from which sources?",
    gloss: "District and source breakdown",
    needs: { disease: "malaria" },
  },
  {
    slot: "audit",
    code: "en",
    text: "Can the EpiClim history train an Odisha outbreak forecast?",
    gloss: "Data audit question · answered from the frozen audit, not from retrieval",
    needs: null,
  },
];

export interface EvidenceRow {
  district?: string | null;
  disease?: string | null;
}

function key(value: string | null | undefined): string {
  return (value ?? "").trim().toLowerCase().replaceAll(" ", "_");
}

/** True when at least one retained record carries the district and disease asked for. */
export function conditionHolds(rows: EvidenceRow[], needs: EvidenceCondition | null): boolean {
  if (!needs) return true;
  const wantDistrict = needs.district ? key(needs.district) : null;
  const wantDisease = needs.disease ? key(needs.disease) : null;
  return rows.some((row) => (
    (wantDistrict === null || key(row.district) === wantDistrict)
    && (wantDisease === null || key(row.disease) === wantDisease)
  ));
}

/**
 * At most one prompt per slot, in slot order. An empty retained set yields only
 * the prompts that need no evidence.
 */
export function availableExamples(
  rows: EvidenceRow[],
  candidates: ExamplePrompt[] = EXAMPLE_CANDIDATES,
): ExamplePrompt[] {
  const chosen = new Map<ExampleSlot, ExamplePrompt>();
  for (const candidate of candidates) {
    if (chosen.has(candidate.slot)) continue;
    if (conditionHolds(rows, candidate.needs)) chosen.set(candidate.slot, candidate);
  }
  return SLOT_ORDER.flatMap((slot) => {
    const prompt = chosen.get(slot);
    return prompt ? [prompt] : [];
  });
}
