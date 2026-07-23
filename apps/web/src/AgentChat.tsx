import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, Eraser, Fingerprint, MessageSquare, Quote } from "lucide-react";
import { api, ApiUnavailable } from "./api";
import { answerCounts, countsLabel, evidenceDigest } from "./agentAnswer";
import { isSynthetic, typedState } from "./epistemics";
import { availableExamples } from "./examples";
import { humanizeDisease } from "./format";
import { LANGUAGES, detectScript, isAnswerLanguage, langAttribute } from "./languages";
import type { AnswerLanguage } from "./languages";
import { RoutingBar, ScriptSelector, TranslateEvidence } from "./Translate";
import { Chip, Notice, SyntheticBadge, TypedState } from "./ui";
import type { AgentEvidenceCitation, AgentHistoryTurn, AgentQueryResult, Signal } from "./types";

/* ------------------------------------------------------------------- helpers */

interface AskedTurn {
  kind: "asked";
  id: string;
  text: string;
  target: AnswerLanguage;
}

interface AnsweredTurn {
  kind: "answered";
  id: string;
  result: AgentQueryResult;
  requestedLanguage: AnswerLanguage;
  clientLatencyMs: number;
}

interface FailedTurn {
  kind: "failed";
  id: string;
  message: string;
}

type Turn = AskedTurn | AnsweredTurn | FailedTurn;

function turnId(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

export function districtLabel(id: string | null | undefined, names: Map<string, string>): string {
  if (!id) return "not resolved";
  return names.get(id.toLowerCase()) ?? id.replace(/^OD-DIST-/i, "").replaceAll("-", " ");
}

/**
 * Splits an answer on `[n]` citation markers so each one can be rendered as a
 * link to the evidence card it points at. Answers without markers pass through
 * untouched — the interface never manufactures a citation that the model did not
 * actually write.
 */
function withCitationMarkers(answer: string, evidenceCount: number, anchorPrefix: string) {
  // Whitespace inside the brackets is tolerated: translation detokenisation
  // routinely turns "[1]" into "[ 1 ]", and a citation must survive that.
  const parts = answer.split(/(\[\s*E?\s*\d{1,2}\s*\])/gi);
  if (parts.length === 1) return answer;
  return parts.map((part, index) => {
    const match = /^\[\s*E?\s*(\d{1,2})\s*\]$/i.exec(part);
    if (!match) return <span key={index}>{part}</span>;
    const number = Number(match[1]);
    if (number < 1 || number > evidenceCount) return <span key={index}>{part}</span>;
    return (
      <a key={index} className="cite-mark" href={`#${anchorPrefix}-${number}`}>
        {String(number).padStart(2, "0")}
      </a>
    );
  });
}

/* ------------------------------------------------------------------ evidence */

function EvidenceCard({
  citation, index, anchor, districtNames, cited,
}: {
  citation: AgentEvidenceCitation;
  index: number;
  anchor: string;
  districtNames: Map<string, string>;
  /** True when the answer text actually references this record. */
  cited: boolean;
}) {
  // Not `content_sha256`: on a non-retention deployment that is the digest of the
  // same placeholder on every row, so it identifies the policy, not the record.
  const digest = evidenceDigest(citation);
  const evidenceText = citation.redacted_evidence ?? "";
  const script = evidenceText ? detectScript(evidenceText) : "und";
  const link = citation.canonical_url ?? null;
  return (
    <article className={`ev${cited ? " ev--cited" : ""}`} id={anchor}>
      <div className="ev__rail" aria-hidden="true">{String(index + 1).padStart(2, "0")}</div>
      <div className="ev__body">
        <div className="ev__top">
          <strong className="ev__source">{citation.source_id ?? "source id not exposed"}</strong>
          <span className="ev__district">{districtLabel(citation.district_id, districtNames)}</span>
          <Chip tone={citation.assertion === "affirmed" ? "ok" : citation.assertion ? "warn" : "mute"} size="sm">
            {citation.assertion?.replaceAll("_", " ") ?? "assertion not stated"}
          </Chip>
          <Chip tone={cited ? "ok" : "mute"} size="sm">{cited ? "cited in answer" : "retrieved, not cited"}</Chip>
          {isSynthetic(citation) && <SyntheticBadge />}
        </div>
        <blockquote lang={langAttribute(script)}>
          <Quote size={13} strokeWidth={3} aria-hidden="true" />
          {evidenceText || "No redacted evidence span was returned for this citation."}
        </blockquote>
        <dl className="ev__kv">
          <div><dt>Disease</dt><dd>{humanizeDisease(citation.disease) ?? "unresolved"}</dd></div>
          <div><dt>Retrieved</dt><dd>{citation.retrieved_at ?? "not exposed"}</dd></div>
          <div><dt>Review</dt><dd>{citation.review_state ?? "unreviewed"}</dd></div>
          <div>
            <dt>{digest.label}</dt>
            <dd><code title={digest.title ?? undefined}>{digest.value}</code></dd>
          </div>
        </dl>
        {link && (
          <p className="ev__link">
            <a href={link} target="_blank" rel="noreferrer">{link}</a>
          </p>
        )}
        {evidenceText && <TranslateEvidence text={evidenceText} sourceLanguage={script} id={anchor} />}
      </div>
    </article>
  );
}

/* --------------------------------------------------------------------- turns */

function AnswerTurn({ turn, districtNames }: { turn: AnsweredTurn; districtNames: Map<string, string> }) {
  const { result } = turn;
  const copy = typedState(result.answer_state);
  const anchorPrefix = `${turn.id}-cite`;
  const answerLanguage = isAnswerLanguage(result.answer_language) ? result.answer_language : undefined;
  const rendered = answerLanguage ?? detectScript(result.answer);
  const languageHonoured = answerLanguage === undefined || answerLanguage === turn.requestedLanguage;
  const generation = result.generation_mode ?? null;
  const [showOriginal, setShowOriginal] = useState(false);
  // Retrieved is not cited: the client asks for up to eight records and the API
  // returns all of them, whatever the answer text went on to reference.
  const counts = answerCounts(result);

  return (
    <article className="turn turn--answer" aria-label="Assistant answer">
      <RoutingBar
        inLanguage={result.scope.question_language}
        outLanguage={result.answer_language ?? rendered}
        model={result.model ?? null}
        note={
          generation === "generated"
            ? "generated from retrieved evidence"
            : generation
              ? String(generation).replaceAll("_", " ")
              : "generation mode not reported"
        }
        latencyMs={result.latency_ms ?? turn.clientLatencyMs}
      />

      <div className="turn__state">
        <Chip tone={copy.tone} size="sm">{copy.label}</Chip>
        <span>{result.intent.replaceAll("_", " ")}</span>
        <span>{countsLabel(counts)}</span>
        {counts.considered !== null && (
          <span className="turn__considered">{counts.considered} scored by the retriever</span>
        )}
      </div>

      <p className="turn__answer" lang={langAttribute(rendered)}>
        {withCitationMarkers(result.answer, result.evidence.length, anchorPrefix)}
      </p>

      {!languageHonoured && (
        <p className="turn__degrade">
          Requested {LANGUAGES[turn.requestedLanguage].exonym}; the API answered in{" "}
          {answerLanguage ? LANGUAGES[answerLanguage].exonym : "an unstated language"}. The text above is shown exactly
          as returned.
        </p>
      )}

      {result.answer_original && result.answer_original !== result.answer && (
        <div className="turn__original">
          <button type="button" className="minibtn" aria-expanded={showOriginal} onClick={() => setShowOriginal(!showOriginal)}>
            {showOriginal ? "Hide" : "Show"} the answer before translation
          </button>
          {showOriginal && <p lang="en">{result.answer_original}</p>}
        </div>
      )}

      <dl className="turn__scope">
        <div><dt>District</dt><dd>{districtLabel(result.scope.district_id, districtNames)}</dd></div>
        <div><dt>Disease</dt><dd>{humanizeDisease(result.scope.disease) ?? "not narrowed"}</dd></div>
      </dl>

      {result.reason_codes.length > 0 && (
        <div className="turn__codes">
          <span className="eyebrow">Reason codes</span>
          <div>{result.reason_codes.map((code) => <code key={code}>{code}</code>)}</div>
        </div>
      )}

      {result.evidence.length > 0 ? (
        <div className="turn__evidence">
          <p className="side-count">
            <span><Fingerprint size={13} strokeWidth={3} aria-hidden="true" /> Records retrieved for this answer</span>
            <strong>{counts.retrieved}</strong>
          </p>
          <p className="turn__evnote">
            {counts.cited === null
              ? "The API did not report which of these the answer cites, so none is marked as cited."
              : `${counts.cited} of ${counts.retrieved} are referenced by the answer text; the rest were retrieved and are shown so the unused evidence stays visible.`}
          </p>
          {result.evidence.map((citation, index) => (
            <EvidenceCard
              key={citation.signal_id ?? `${citation.source_snapshot_id ?? "cite"}-${index}`}
              citation={citation}
              index={index}
              anchor={`${anchorPrefix}-${index + 1}`}
              districtNames={districtNames}
              cited={Boolean(citation.signal_id && counts.citedIds.has(citation.signal_id))}
            />
          ))}
        </div>
      ) : (
        <TypedState code={result.answer_state} capability="agent_query" compact />
      )}
    </article>
  );
}

function WorkingTurn({ startedAt, target }: { startedAt: number; target: AnswerLanguage }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setElapsed(Date.now() - startedAt), 120);
    return () => window.clearInterval(timer);
  }, [startedAt]);
  return (
    <article className="turn turn--working" aria-live="polite" aria-label="Working">
      <div className="working">
        <span className="working__caret" aria-hidden="true" />
        <span className="working__text">
          Retrieving evidence and writing an answer in {LANGUAGES[target].exonym}. On CPU this takes seconds, not
          milliseconds; nothing is shown until the API returns.
        </span>
        <span className="working__ms">{(elapsed / 1000).toFixed(1)}s</span>
      </div>
      <code className="working__call">POST /api/v1/agent/query</code>
    </article>
  );
}

/* ---------------------------------------------------------------------- page */

export function AgentChat({
  districtNames, signals,
}: {
  districtNames: Map<string, string>;
  /** Retained evidence, used to offer only starter prompts that have records. */
  signals: Signal[];
}) {
  const [thread, setThread] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [target, setTarget] = useState<AnswerLanguage>("en");
  const [pending, setPending] = useState<{ startedAt: number; target: AnswerLanguage } | null>(null);
  const [lastModel, setLastModel] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Only follow the query log once it has started; on an empty log this
    // would scroll the page heading out of view before the operator has read it.
    if (!thread.length && !pending) return;
    endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [thread.length, pending]);

  /**
   * Starter prompts are filtered against the evidence actually retained, so an
   * example never demonstrates the agent by returning nothing.
   */
  const examples = useMemo(() => availableExamples(signals), [signals]);

  const history = useMemo<AgentHistoryTurn[]>(() => {
    const turns: AgentHistoryTurn[] = [];
    for (const turn of thread) {
      if (turn.kind === "asked") turns.push({ role: "user", content: turn.text });
      if (turn.kind === "answered") turns.push({ role: "assistant", content: turn.result.answer });
    }
    return turns.slice(-8);
  }, [thread]);

  const ask = async (value: string) => {
    const question = value.trim();
    if (question.length < 3 || pending) return;
    const asked: AskedTurn = { kind: "asked", id: turnId("q"), text: question, target };
    setThread((current) => [...current, asked]);
    setDraft("");
    const startedAt = Date.now();
    setPending({ startedAt, target });
    try {
      const response = await api.agentQuery({
        question,
        maximum_evidence: 8,
        target_language: target,
        ...(history.length ? { history } : {}),
      });
      const result = response.data;
      setLastModel(result.model ?? null);
      setThread((current) => [
        ...current,
        {
          kind: "answered",
          id: turnId("a"),
          result,
          requestedLanguage: target,
          clientLatencyMs: Date.now() - startedAt,
        },
      ]);
    } catch (error) {
      const status = error instanceof ApiUnavailable ? error.status : undefined;
      setThread((current) => [
        ...current,
        {
          kind: "failed",
          id: turnId("f"),
          message:
            status === undefined
              ? "The request never reached the evidence API. No answer was drafted in the browser in its place."
              : `The evidence API answered ${status}. No answer was drafted in its place.`,
        },
      ]);
    } finally {
      setPending(null);
      window.requestAnimationFrame(() => inputRef.current?.focus());
    }
  };

  const draftScript = detectScript(draft);

  return (
    <section className="page-section chat">
      <div className="page-head">
        <div>
          <span className="eyebrow">Source-grounded agent · Odia · Hindi · English</span>
          <h1>Public-health evidence agent</h1>
          <p>
            Ask what the registered sources published about a disease or district. Returned records carry their source,
            district and evidence span; questions without supporting evidence produce an explicit refusal.
          </p>
        </div>
        <div className="page-head__status">
          <Chip tone={lastModel ? "ok" : "mute"}>{lastModel ?? "model not yet reported"}</Chip>
        </div>
      </div>

      <div className="chat__deck">
        <ScriptSelector value={target} onChange={setTarget} legend="Answer language" idPrefix="chat-target" compact />
        <div className="chat__deckside">
          <p className="chat__decknote">
            Follow-up questions retain the last eight turns. The selector controls the answer language.
          </p>
          {thread.length > 0 && (
            <button type="button" className="minibtn" onClick={() => setThread([])} disabled={Boolean(pending)}>
              <Eraser size={13} strokeWidth={3} aria-hidden="true" /> Clear results
            </button>
          )}
        </div>
      </div>

      <div className="chat__thread" role="log" aria-label="Evidence-agent query results" aria-busy={Boolean(pending)}>
        {thread.length === 0 && !pending && (
          <div className="chat__intro">
            <MessageSquare size={30} strokeWidth={2.5} aria-hidden="true" />
            <h2>Start with a district and a disease</h2>
            <p>
              The assistant answers from records this platform has actually retrieved. It counts published documents; it
              never reports case numbers, and it refuses clinical advice outright.
            </p>
            <div className="chat__examples">
              {examples.map((example) => (
                <button key={example.text} type="button" onClick={() => void ask(example.text)}>
                  <span className="chat__examplemark" lang={example.code} aria-hidden="true">{LANGUAGES[example.code].mark}</span>
                  <span>
                    <b lang={example.code}>{example.text}</b>
                    <small>{example.gloss}</small>
                  </span>
                </button>
              ))}
            </div>
            <p className="chat__examplenote">
              {examples.length
                ? "Each example is offered only while a retained record matches its district and disease tag, so it demonstrates the agent instead of demonstrating an empty result."
                : "No starter prompt is offered: nothing retained matches one yet. Ask about any district and disease and the answer will state what was and was not found."}
            </p>
          </div>
        )}

        {thread.map((turn) => {
          if (turn.kind === "asked") {
            const script = detectScript(turn.text);
            return (
              <article key={turn.id} className="turn turn--asked" aria-label="Your question">
                <span className="turn__who" lang={langAttribute(script)} aria-hidden="true">
                  {isAnswerLanguage(script) ? LANGUAGES[script].mark : "?"}
                </span>
                <p lang={langAttribute(script)}>{turn.text}</p>
                <span className="turn__want">answer in {LANGUAGES[turn.target].exonym}</span>
              </article>
            );
          }
          if (turn.kind === "failed") {
            return (
              <div key={turn.id} className="turn turn--failed">
                <Notice tone="stop" title="No answer was produced">
                  <p>{turn.message}</p>
                </Notice>
              </div>
            );
          }
          return <AnswerTurn key={turn.id} turn={turn} districtNames={districtNames} />;
        })}

        {pending && <WorkingTurn startedAt={pending.startedAt} target={pending.target} />}
        <div ref={endRef} />
      </div>

      <form
        // The composer only pins to the viewport once there is a query log to
        // follow; on the empty state it would sit on top of the example prompts.
        className={`chat__composer${thread.length || pending ? " chat__composer--stuck" : ""}`}
        onSubmit={(event) => { event.preventDefault(); void ask(draft); }}
      >
        <label htmlFor="chat-input" className="sr-only">Your question, in Odia, Hindi or English</label>
        <textarea
          id="chat-input"
          ref={inputRef}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void ask(draft);
            }
          }}
          placeholder="ପ୍ରଶ୍ନ ଲେଖନ୍ତୁ · सवाल लिखें · ask a question"
          rows={2}
          maxLength={500}
          lang={langAttribute(draftScript)}
          disabled={Boolean(pending)}
        />
        <button type="submit" disabled={Boolean(pending) || draft.trim().length < 3} aria-label="Send question">
          <ArrowUp strokeWidth={3} aria-hidden="true" />
        </button>
        <p className="chat__hint">
          Enter sends · Shift+Enter adds a line · {draft.length}/500 · sent only to the configured evidence API
        </p>
      </form>

      <Notice tone="warn" title="Decision-support boundary">
        <p>
          This assistant searches the collected public evidence and can rank current weather-related review priority.
          It does not diagnose people, prescribe treatment, or autonomously dispatch an alert; verified public-health
          staff make those decisions. A calibrated district disease probability becomes available only after an
          authorised surveillance-data and validation phase.
        </p>
      </Notice>
    </section>
  );
}
