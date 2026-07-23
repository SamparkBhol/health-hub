import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeftRight, ArrowRight, Check, Copy, Languages } from "lucide-react";
import { api, ApiUnavailable } from "./api";
import { LANGUAGES, LANGUAGE_ORDER, detectScript, isAnswerLanguage, langAttribute, scriptTag } from "./languages";
import type { AnswerLanguage } from "./languages";
import { Chip, Notice, SyntheticBadge, TypedState } from "./ui";
import type { TranslateRequest, TranslationResult } from "./types";

/* ------------------------------------------------------------------ machinery */

export type TranslateState =
  | { phase: "idle" }
  | { phase: "working"; target: AnswerLanguage }
  | { phase: "done"; result: TranslationResult }
  | { phase: "unavailable"; code: string; detail: string };

const MISSING_ROUTE = "translation_unavailable_source_language_only";

/**
 * Calls the translation service and turns every failure into a typed state.
 *
 * A 404 means this deployment has not mounted the translation route at all,
 * which is a capability statement rather than an error: the interface says the
 * source language is all it can offer, and never substitutes machine output of
 * its own.
 */
export async function runTranslation(
  text: string,
  target: AnswerLanguage,
  source: AnswerLanguage | "auto",
): Promise<TranslateState> {
  try {
    const request: TranslateRequest = {
      text,
      target_language: target,
      ...(source === "auto" ? {} : { source_language: source }),
    };
    const response = await api.translate(request);
    const result = response.data;
    if (result.status === "unavailable" || !result.translated_text) {
      return {
        phase: "unavailable",
        code: result.capability_code ?? MISSING_ROUTE,
        detail: "The translation service answered but returned no text for this pair.",
      };
    }
    return { phase: "done", result };
  } catch (error) {
    const status = error instanceof ApiUnavailable ? error.status : undefined;
    if (status === 404 || status === 501) {
      return {
        phase: "unavailable",
        code: MISSING_ROUTE,
        detail: "POST /api/v1/translate is not mounted in this deployment, so evidence stays in its source language.",
      };
    }
    return {
      phase: "unavailable",
      code: status === 503 ? "capacity_exceeded" : "unknown",
      detail:
        status === undefined
          ? "The translation request did not reach the API. Nothing was translated in the browser instead."
          : `The translation service answered ${status}. No text was generated in its place.`,
    };
  }
}

/* --------------------------------------------------------------- shared parts */

/**
 * The signature control. Each language is a hard block carrying one glyph from
 * its own script — the fastest possible read for an operator who cannot read
 * all three — with the endonym under it and the FLORES tag the model is given.
 */
export function ScriptSelector({
  value, onChange, legend, exclude = [], idPrefix, compact = false,
}: {
  value: AnswerLanguage;
  onChange: (next: AnswerLanguage) => void;
  legend: string;
  exclude?: AnswerLanguage[];
  idPrefix: string;
  compact?: boolean;
}) {
  const options = LANGUAGE_ORDER.filter((code) => !exclude.includes(code));
  return (
    <fieldset className={`scriptpick${compact ? " scriptpick--compact" : ""}`}>
      <legend>{legend}</legend>
      <div className="scriptpick__row">
        {options.map((code) => {
          const spec = LANGUAGES[code];
          const id = `${idPrefix}-${code}`;
          return (
            <label key={code} className={`scriptpick__opt${value === code ? " is-on" : ""}`} htmlFor={id}>
              <input
                id={id}
                type="radio"
                name={idPrefix}
                value={code}
                checked={value === code}
                onChange={() => onChange(code)}
              />
              <span className="scriptpick__mark" lang={code} aria-hidden="true">{spec.mark}</span>
              <span className="scriptpick__text">
                <b lang={code}>{spec.endonym}</b>
                <small>{spec.tag}</small>
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

/**
 * IN → MACHINE → OUT. The same three-block strip appears above every generated
 * answer and above every translation, because in a trilingual product the single
 * most load-bearing fact is which language went in, what handled it, and which
 * language came out.
 */
export function RoutingBar({
  inLanguage, outLanguage, model, note, latencyMs,
}: {
  inLanguage: string | null | undefined;
  outLanguage: string | null | undefined;
  model: string | null | undefined;
  note?: string;
  latencyMs?: number | null;
}) {
  return (
    <div className="routing" role="group" aria-label="Language routing">
      <span className="routing__cell">
        <small>In</small>
        <b lang={langAttribute(inLanguage)}>{isAnswerLanguage(inLanguage) ? LANGUAGES[inLanguage].endonym : "undetermined"}</b>
        <code>{scriptTag(inLanguage)}</code>
      </span>
      <ArrowRight size={15} strokeWidth={3.5} aria-hidden="true" />
      <span className="routing__cell routing__cell--machine">
        <small>Machine</small>
        <b>{model ?? "not reported"}</b>
        <code>{note ?? (model ? "model id as returned" : "no model id in payload")}</code>
      </span>
      <ArrowRight size={15} strokeWidth={3.5} aria-hidden="true" />
      <span className="routing__cell">
        <small>Out</small>
        <b lang={langAttribute(outLanguage)}>{isAnswerLanguage(outLanguage) ? LANGUAGES[outLanguage].endonym : "undetermined"}</b>
        <code>{scriptTag(outLanguage)}</code>
      </span>
      {typeof latencyMs === "number" && <span className="routing__ms">{Math.round(latencyMs)} ms</span>}
    </div>
  );
}

export function PipelineTrace({ stages }: { stages: string[] }) {
  if (!stages.length) return null;
  return (
    <ol className="trace" aria-label="Translation pipeline stages">
      {stages.map((stage, index) => (
        <li key={`${stage}-${index}`} className="trace__stage"><code>{stage}</code></li>
      ))}
    </ol>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return undefined;
    const timer = window.setTimeout(() => setCopied(false), 1600);
    return () => window.clearTimeout(timer);
  }, [copied]);
  const copy = () => {
    void navigator.clipboard?.writeText(text).then(() => setCopied(true)).catch(() => setCopied(false));
  };
  return (
    <button type="button" className="minibtn" onClick={copy}>
      {copied ? <Check size={13} strokeWidth={3} aria-hidden="true" /> : <Copy size={13} strokeWidth={3} aria-hidden="true" />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

/* ------------------------------------------------ per-record translate control */

/**
 * The per-evidence affordance. An English-reading reviewer presses ଓଡ଼ିଆ or
 * हिन्दी and gets the record in that script; an Odia reader presses English.
 * The original is never replaced — the translation is appended below it, tagged,
 * so the evidence of record stays the source text.
 */
export function TranslateEvidence({
  text, sourceLanguage, id,
}: {
  text: string;
  sourceLanguage?: string | null;
  id: string;
}) {
  const [state, setState] = useState<TranslateState>({ phase: "idle" });
  const detected = isAnswerLanguage(sourceLanguage) ? sourceLanguage : detectScript(text);
  const exclude = isAnswerLanguage(detected) ? [detected] : [];

  const request = async (target: AnswerLanguage) => {
    setState({ phase: "working", target });
    setState(await runTranslation(text, target, isAnswerLanguage(detected) ? detected : "auto"));
  };

  return (
    <div className="xlate">
      <div className="xlate__bar">
        <span className="xlate__label"><Languages size={13} strokeWidth={3} aria-hidden="true" /> Read this in</span>
        {LANGUAGE_ORDER.filter((code) => !exclude.includes(code)).map((code) => (
          <button
            key={code}
            type="button"
            className="xlate__btn"
            disabled={state.phase === "working"}
            onClick={() => void request(code)}
            aria-label={`Translate this evidence into ${LANGUAGES[code].exonym}`}
          >
            <span lang={code} aria-hidden="true">{LANGUAGES[code].mark}</span>
            <b lang={code}>{LANGUAGES[code].endonym}</b>
          </button>
        ))}
        {state.phase === "done" && (
          <button type="button" className="xlate__btn xlate__btn--clear" onClick={() => setState({ phase: "idle" })}>
            Hide translation
          </button>
        )}
      </div>

      {state.phase === "working" && (
        <p className="xlate__pending" aria-live="polite">
          Translating into {LANGUAGES[state.target].exonym}. The source text above is unchanged.
        </p>
      )}

      {state.phase === "unavailable" && (
        <TypedState code={state.code} capability="machine_translation" detail={state.detail} compact />
      )}

      {state.phase === "done" && (
        <div className="xlate__out" id={`${id}-translation`}>
          <div className="xlate__outhead">
            <Chip tone="ok" size="sm">Machine translation</Chip>
            <code>{scriptTag(state.result.source_language)} → {scriptTag(state.result.target_language)}</code>
            <CopyButton text={state.result.translated_text ?? ""} />
          </div>
          <p className="xlate__text" lang={langAttribute(state.result.target_language)}>
            {state.result.translated_text}
          </p>
          <PipelineTrace stages={state.result.pipeline ?? []} />
          <p className="xlate__model">
            {state.result.model ?? "model not reported"} · not a certified translation; the source span above remains the
            record of evidence
          </p>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------- the page */

const SAMPLES: Array<{ code: AnswerLanguage; text: string; note: string }> = [
  { code: "en", text: "Dengue cases were reported in the district this week.", note: "English notice line" },
  { code: "hi", text: "जिले में डेंगू के मामले सामने आए।", note: "Hindi bulletin line" },
  { code: "or", text: "ଖୋର୍ଦ୍ଧା ଜିଲ୍ଲାରେ ଡେଙ୍ଗୁ ମାମଲା ଚିହ୍ନଟ ହୋଇଛି।", note: "Odia circular line" },
];

export function TranslationPage() {
  const [source, setSource] = useState<AnswerLanguage | "auto">("auto");
  const [target, setTarget] = useState<AnswerLanguage>("or");
  const [text, setText] = useState("");
  const [state, setState] = useState<TranslateState>({ phase: "idle" });
  const outputRef = useRef<HTMLDivElement>(null);

  const detected = useMemo(() => detectScript(text), [text]);
  const effectiveSource = source === "auto" ? detected : source;

  const submit = useCallback(async (value: string, to: AnswerLanguage, from: AnswerLanguage | "auto") => {
    const trimmed = value.trim();
    if (!trimmed) return;
    setState({ phase: "working", target: to });
    setState(await runTranslation(trimmed, to, from));
  }, []);

  useEffect(() => {
    if (state.phase === "done") outputRef.current?.focus();
  }, [state.phase]);

  const swap = () => {
    if (state.phase === "done" && state.result.translated_text) {
      const nextSource = state.result.target_language;
      const nextTarget = isAnswerLanguage(state.result.source_language) ? state.result.source_language : "en";
      setText(state.result.translated_text);
      setSource(nextSource);
      setTarget(nextTarget);
      setState({ phase: "idle" });
      return;
    }
    if (isAnswerLanguage(effectiveSource)) {
      const nextTarget = effectiveSource;
      setSource(target);
      setTarget(nextTarget);
    }
  };

  const pairLabel = `${scriptTag(effectiveSource)} → ${scriptTag(target)}`;
  const sameLanguage = effectiveSource === target;

  return (
    <section className="page-section page-section--narrow xpage">
      <div className="page-head">
        <div>
          <span className="eyebrow">Odia · Hindi · English</span>
          <h1>Translation workspace</h1>
          <p>
            Put a notice, a headline or an evidence span in one box and read it in another script. The service runs on
            this deployment&rsquo;s own CPU; nothing is sent to a third-party translation vendor.
          </p>
        </div>
        <Chip tone="mute">POST /api/v1/translate</Chip>
      </div>

      <div className="xgrid">
        <div className="xpane">
          <div className="xpane__head">
            <span className="eyebrow">Source text</span>
            <div className="xpane__src">
              <label htmlFor="xlate-source">Source language</label>
              <select
                id="xlate-source"
                value={source}
                onChange={(event) => setSource(event.target.value === "auto" ? "auto" : (event.target.value as AnswerLanguage))}
              >
                <option value="auto">
                  Detect from script{isAnswerLanguage(detected) ? ` — ${LANGUAGES[detected].exonym}` : ""}
                </option>
                {LANGUAGE_ORDER.map((code) => (
                  <option key={code} value={code}>{LANGUAGES[code].exonym} — {LANGUAGES[code].endonym}</option>
                ))}
              </select>
            </div>
          </div>
          <textarea
            className="xpane__input"
            aria-label="Text to translate"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="ଏଠାରେ ଲେଖନ୍ତୁ · यहाँ लिखें · type here"
            rows={7}
            maxLength={2000}
            lang={langAttribute(effectiveSource)}
          />
          <div className="xpane__foot">
            <span>{text.length}/2000 · script detected: {detected === "und" ? "none yet" : detected}</span>
            <div className="xpane__samples">
              {SAMPLES.map((sample) => (
                <button
                  key={sample.code}
                  type="button"
                  onClick={() => { setText(sample.text); setSource(sample.code); setTarget(sample.code === "en" ? "or" : "en"); setState({ phase: "idle" }); }}
                >
                  <span lang={sample.code} aria-hidden="true">{LANGUAGES[sample.code].mark}</span>
                  {sample.note}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="xswap">
          <code>{pairLabel}</code>
          <button type="button" onClick={swap} aria-label="Swap source and target languages">
            <ArrowLeftRight size={18} strokeWidth={3} aria-hidden="true" />
          </button>
        </div>

        <div className="xpane xpane--out">
          <div className="xpane__head">
            <span className="eyebrow">Translated output</span>
          </div>
          <ScriptSelector
            value={target}
            onChange={(next) => { setTarget(next); setState({ phase: "idle" }); }}
            legend="Translate into"
            idPrefix="xlate-target"
          />

          <div className="xpane__output" ref={outputRef} tabIndex={-1} aria-live="polite" aria-busy={state.phase === "working"}>
            {state.phase === "idle" && (
              <p className="xpane__idle">
                {sameLanguage && text.trim()
                  ? "Source and target are the same language. Pick a different output language."
                  : "Press Translate. The output appears here in the target script, with the model and the exact pipeline it used."}
              </p>
            )}
            {state.phase === "working" && (
              <p className="pending pending--block">Running the translation model on CPU. Nothing is drafted while this waits.</p>
            )}
            {state.phase === "unavailable" && (
              <TypedState code={state.code} capability="machine_translation" detail={state.detail} />
            )}
            {state.phase === "done" && (
              <>
                <p className="xpane__text" lang={langAttribute(state.result.target_language)}>
                  {state.result.translated_text}
                </p>
                <div className="xpane__actions">
                  <CopyButton text={state.result.translated_text ?? ""} />
                  {state.result.is_synthetic && <SyntheticBadge label="Synthetic" />}
                </div>
              </>
            )}
          </div>

          <button
            type="button"
            className="btn btn--primary xpane__go"
            disabled={state.phase === "working" || !text.trim() || sameLanguage}
            onClick={() => void submit(text, target, source)}
          >
            Translate <ArrowRight size={17} strokeWidth={3} aria-hidden="true" />
          </button>
        </div>
      </div>

      {state.phase === "done" && (
        <div className="xreceipt">
          <RoutingBar
            inLanguage={state.result.source_language}
            outLanguage={state.result.target_language}
            model={state.result.model}
            note={state.result.source_language_detected ? "source language detected" : "source language declared"}
            latencyMs={state.result.latency_ms}
          />
          <div className="xreceipt__trace">
            <span className="eyebrow">Pipeline actually executed</span>
            <PipelineTrace stages={state.result.pipeline ?? []} />
          </div>
        </div>
      )}

      <Notice tone="warn" title="What machine translation here is and is not">
        <p>
          Output is unreviewed machine translation for comprehension. It is not a certified translation, it is not an
          official communication, and it must not be republished as the source organisation&rsquo;s words. The source span
          always stays on screen next to it, because the source is the evidence and the translation is a reading aid.
        </p>
      </Notice>
    </section>
  );
}
