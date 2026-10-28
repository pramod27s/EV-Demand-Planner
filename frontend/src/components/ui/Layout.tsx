import type { ReactNode } from "react";
import type { IconName } from "./Icon";

/** Page title for the Model Lab pages. (`eyebrow` is kept for older call sites.) */
export function PageHeader({ title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <div className="border-b border-zinc-200 pb-4">
      <h1 className="text-xl font-semibold text-zinc-900">{title}</h1>
      {children && <p className="mt-1 max-w-3xl text-sm leading-relaxed text-zinc-600">{children}</p>}
    </div>
  );
}

/** Bordered panel with a title row and optional actions. (`icon` is accepted but not drawn.) */
export function Section({ title, sub, actions, children, className = "" }: {
  title?: string; sub?: string; icon?: IconName; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-zinc-200 px-4 py-2.5">
          <div>
            {title && <h2 className="text-sm font-semibold text-zinc-900">{title}</h2>}
            {sub && <p className="text-xs text-zinc-500">{sub}</p>}
          </div>
          {actions}
        </div>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

/** Compact segmented control. Options may carry a colour swatch (e.g. optimiser colour). */
export function Segmented<T extends string | number>({ options, value, onChange, label }: {
  options: { value: T; label: string; color?: string }[]; value: T; onChange: (v: T) => void; label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap rounded border border-zinc-300 bg-white p-0.5">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button key={String(o.value)} role="radio" aria-checked={active} onClick={() => onChange(o.value)}
            className={`flex items-center gap-1.5 rounded-sm px-2.5 py-1 text-xs font-medium transition-colors ${
              active ? "bg-zinc-900 text-white" : "text-zinc-600 hover:bg-zinc-100"}`}>
            {o.color && <span className="h-2 w-2 rounded-full" style={{ background: o.color }} />}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Badge({ children, tone = "green" }: { children: ReactNode; tone?: "green" | "amber" | "slate" }) {
  const tones = {
    green: "border-green-300 text-green-800",
    amber: "border-amber-300 text-amber-800",
    slate: "border-zinc-300 text-zinc-600",
  };
  return <span className={`inline-flex items-center rounded-sm border px-1.5 py-0.5 text-2xs font-medium ${tones[tone]}`}>{children}</span>;
}

/** Label / value row used in the side panel. */
export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className="text-right">
        <span className="num text-sm font-medium text-zinc-900">{value}</span>
        {sub && <span className="block text-2xs text-zinc-500">{sub}</span>}
      </span>
    </div>
  );
}
