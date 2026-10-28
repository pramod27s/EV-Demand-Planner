import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_TICK, MODEL_COLOR, TOOLTIP_STYLE } from "../../theme";
import type { ComparisonRow } from "../../types";

const SHOWN = ["Linear Regression", "Random Forest", "XGBoost (default)", "XGBoost + GA", "XGBoost + PSO", "XGBoost + SMA"];
const METRICS: { key: "rmse" | "mae" | "r2"; title: string }[] = [
  { key: "rmse", title: "RMSE (kWh) — lower is better" },
  { key: "mae", title: "MAE (kWh) — lower is better" },
  { key: "r2", title: "R² — higher is better" },
];

const short = (m: string) => m.replace("XGBoost", "XGB").replace("Linear Regression", "LR").replace("Random Forest", "RF");

/** One small bar chart per metric (never a dual axis). Y-axes are zoomed to show differences. */
export default function ComparisonBarChart({ rows }: { rows: ComparisonRow[] }) {
  const data = SHOWN.map((m) => rows.find((r) => r.model === m)).filter((r): r is ComparisonRow => !!r);
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {METRICS.map(({ key, title }) => {
        const vals = data.map((d) => d[key]);
        const lo = Math.min(...vals);
        const hi = Math.max(...vals);
        const pad = (hi - lo) * 0.5 || 0.01;
        return (
          <div key={key}>
            <div className="mb-1 text-sm font-medium text-slate-700">{title}</div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={data} margin={{ top: 18, right: 4, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#eef2f7" vertical={false} />
                <XAxis dataKey="model" tickFormatter={short} interval={0} angle={-30} textAnchor="end"
                  height={56} tick={AXIS_TICK} tickLine={false} />
                <YAxis domain={[Math.max(0, lo - pad), hi + pad * 0.5]} tickFormatter={(v: number) => v.toFixed(2)}
                  width={44} tick={AXIS_TICK} tickLine={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => v.toFixed(4)} cursor={{ fill: "#f1f5f9" }} />
                <Bar dataKey={key} radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {data.map((d) => <Cell key={d.model} fill={MODEL_COLOR[d.model]} />)}
                  <LabelList dataKey={key} position="top" formatter={(v: number) => v.toFixed(3)}
                    style={{ fontSize: 10, fill: "#1f2937" }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
}
