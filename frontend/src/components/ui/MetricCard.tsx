import type { IconName } from "./Icon";

interface Props {
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  icon?: IconName;     // accepted for older call sites; not drawn
  accent?: string;     // small swatch next to the label
  delta?: { text: string; good: boolean };
}

/** Flat stat tile: label, monospaced value, optional change and note. */
export default function MetricCard({ label, value, unit, sub, accent, delta }: Props) {
  return (
    <div className="card px-4 py-3">
      <div className="flex items-center gap-1.5 text-xs text-zinc-500">
        {accent && <span className="h-2 w-2 rounded-sm" style={{ background: accent }} />}
        {label}
      </div>
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className="num text-2xl font-semibold text-zinc-900">{value}</span>
        {unit && <span className="text-xs text-zinc-500">{unit}</span>}
        {delta && (
          <span className={`num ml-auto text-xs font-medium ${delta.good ? "text-green-700" : "text-red-700"}`}>{delta.text}</span>
        )}
      </div>
      {sub && <div className="mt-0.5 text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}
