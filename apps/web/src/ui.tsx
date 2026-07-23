import { AlertTriangle, CircleSlash } from "lucide-react";
import { typedState } from "./epistemics";
import type { Tone } from "./epistemics";
import type { Warning } from "./types";

export function Chip({ children, tone = "mute", size = "md" }: { children: React.ReactNode; tone?: Tone; size?: "sm" | "md" }) {
  return <span className={`chip chip--${tone} chip--${size}`}>{children}</span>;
}

/**
 * Hard-edged hazard badge. Rendered only from a payload flag, never from prose, so a
 * synthetic value can never reach the screen looking like an observation.
 */
export function SyntheticBadge({ label = "Synthetic" }: { label?: string }) {
  return (
    <span className="synth" role="img" aria-label={`${label}: synthetic test data, not an observation`}>
      <span className="synth__stripes" aria-hidden="true" />
      <span className="synth__text">{label}</span>
    </span>
  );
}

export function Notice({ tone, title, children }: { tone: Tone; title: string; children?: React.ReactNode }) {
  return (
    <div className={`notice notice--${tone}`} role={tone === "stop" ? "alert" : undefined}>
      <AlertTriangle size={20} strokeWidth={2.5} aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        {children}
      </div>
    </div>
  );
}

/** Renders any typed API state as a readable panel. Never a blank box, never a spinner. */
export function TypedState({
  code, reasonCode, capability, compact = false, detail,
}: {
  code: string;
  reasonCode?: string | null;
  capability?: string;
  compact?: boolean;
  /** Deployment-specific sentence explaining this instance of the state. */
  detail?: string | null;
}) {
  const copy = typedState(code);
  return (
    <div className={`typed-state typed-state--${copy.tone}${compact ? " typed-state--compact" : ""}`}>
      <div className="typed-state__head">
        <CircleSlash size={compact ? 18 : 26} strokeWidth={2.5} aria-hidden="true" />
        <Chip tone={copy.tone} size="sm">{copy.label}</Chip>
      </div>
      <h3>{copy.title}</h3>
      <p>{copy.body}</p>
      {detail && <p className="typed-state__detail">{detail}</p>}
      <div className="typed-state__codes">
        {capability && <code>{capability}</code>}
        <code>{code}</code>
        {reasonCode && <code>{reasonCode}</code>}
      </div>
    </div>
  );
}

export function WarningStrip({ warnings }: { warnings: Warning[] }) {
  if (!warnings.length) return null;
  return (
    <ul className="warn-strip">
      {warnings.map((warning) => (
        <li key={warning.code} className={`warn-strip__item warn-strip__item--${warning.severity}`}>
          <span className="warn-strip__code">{warning.code}</span>
          <span className="warn-strip__message">{warning.message}</span>
        </li>
      ))}
    </ul>
  );
}
