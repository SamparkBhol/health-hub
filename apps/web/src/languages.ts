/**
 * The three languages this platform actually reads and writes, and the script
 * facts the interface needs in order to render them correctly.
 *
 * `tag` is the FLORES-200 script tag the translation stack speaks
 * (`eng_Latn` / `hin_Deva` / `ory_Orya`). It is shown in the routing bar because
 * it is the real identifier the model is given, not a decorative label.
 */

export type AnswerLanguage = "en" | "hi" | "or";

export interface LanguageSpec {
  code: AnswerLanguage;
  /** The language's own name, in its own script. */
  endonym: string;
  /** English name, for operators who cannot read the endonym. */
  exonym: string;
  /** One glyph that identifies the script at a glance. */
  mark: string;
  /** FLORES-200 language + script tag used by the translation model. */
  tag: string;
  script: "Latin" | "Devanagari" | "Odia";
}

export const LANGUAGES: Record<AnswerLanguage, LanguageSpec> = {
  en: { code: "en", endonym: "English", exonym: "English", mark: "A", tag: "eng_Latn", script: "Latin" },
  hi: { code: "hi", endonym: "हिन्दी", exonym: "Hindi", mark: "अ", tag: "hin_Deva", script: "Devanagari" },
  or: { code: "or", endonym: "ଓଡ଼ିଆ", exonym: "Odia", mark: "ଓ", tag: "ory_Orya", script: "Odia" },
};

export const LANGUAGE_ORDER: AnswerLanguage[] = ["en", "hi", "or"];

export function isAnswerLanguage(value: string | null | undefined): value is AnswerLanguage {
  return value === "en" || value === "hi" || value === "or";
}

/** FLORES tag for any code the API reports; `und` when the code is not one of the three. */
export function scriptTag(code: string | null | undefined): string {
  return isAnswerLanguage(code) ? LANGUAGES[code].tag : "und";
}

export function languageLabel(code: string | null | undefined): string {
  if (isAnswerLanguage(code)) return `${LANGUAGES[code].endonym} (${LANGUAGES[code].exonym})`;
  if (code === "mixed") return "Mixed script";
  return "Undetermined";
}

const ODIA = /[଀-୿]/;
const DEVANAGARI = /[ऀ-ॿ]/;
const LATIN = /[A-Za-z]/;

/**
 * Which script the characters are in — a deterministic Unicode-block test, not a
 * language classifier. Used only to set `lang`/font so text renders with the right
 * glyphs. The API's own `question_language` is what gets displayed as the routing
 * decision; this never overrides it.
 */
export function detectScript(text: string): AnswerLanguage | "mixed" | "und" {
  const hits: AnswerLanguage[] = [];
  if (ODIA.test(text)) hits.push("or");
  if (DEVANAGARI.test(text)) hits.push("hi");
  if (LATIN.test(text)) hits.push("en");
  if (hits.length === 1) return hits[0];
  if (hits.length > 1) return "mixed";
  return "und";
}

/** A `lang` attribute value, or undefined when no honest one exists. */
export function langAttribute(code: string | null | undefined): string | undefined {
  return isAnswerLanguage(code) ? code : undefined;
}
