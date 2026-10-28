import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_TICK, TOOLTIP_STYLE } from "../../theme";

export interface Series {
  key: string;
  label: string;
  color: string;
  dash?: string;
  values: number[];
}

interface Props {
  series: Series[];
  height?: number;
  /** Show only the first `upTo` iterations (used by the replay animation). */
  upTo?: number;
  yLabel?: string;
}

export default function ConvergenceChart({ series, height = 320, upTo, yLabel = "Validation RMSE (kWh)" }: Props) {
  const n = Math.max(0, ...series.map((s) => s.values.length));
  const rows = Array.from({ length: n }, (_, i) => {
    const row: Record<string, number | null> = { iteration: i + 1 };
    for (const s of series) row[s.key] = upTo !== undefined && i >= upTo ? null : (s.values[i] ?? null);
    return row;
  });
  const all = series.flatMap((s) => s.values);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 0.01;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 4, right: 12, bottom: 16, left: 4 }}>
        <CartesianGrid stroke="#eef2f7" vertical={false} />
        <XAxis dataKey="iteration" tick={AXIS_TICK} allowDecimals={false} tickLine={false} axisLine={{ stroke: "#e2e8f0" }}
          label={{ value: "Iteration", position: "insideBottom", offset: -8, fontSize: 12, fill: "#94a3b8" }} />
        <YAxis domain={[lo - pad, hi + pad]} tickFormatter={(v: number) => v.toFixed(3)} width={64} tick={AXIS_TICK}
          tickLine={false} axisLine={false}
          label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 12, fill: "#94a3b8", dy: 60 }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => v.toFixed(4)} labelFormatter={(l) => `Iteration ${l}`} />
        <Legend verticalAlign="top" align="right" height={32} iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
        {series.map((s) => (
          <Line key={s.key} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2.25}
            strokeDasharray={s.dash} dot={{ r: 3, strokeWidth: 2, fill: "#fff" }} activeDot={{ r: 5 }}
            isAnimationActive={false} connectNulls={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
