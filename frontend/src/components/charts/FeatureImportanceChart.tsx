import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_TICK, COLORS, TOOLTIP_STYLE } from "../../theme";
import type { FeatureImportance } from "../../types";

export default function FeatureImportanceChart({ items }: { items: FeatureImportance[] }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(220, items.length * 26)}>
      <BarChart data={items} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 8 }}>
        <CartesianGrid stroke="#eef2f7" horizontal={false} />
        <XAxis type="number" tick={AXIS_TICK} tickLine={false} />
        <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 11, fill: "#374151" }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => v.toFixed(1)} cursor={{ fill: "#f1f5f9" }} />
        <Bar dataKey="gain" fill={COLORS.sma} radius={[0, 4, 4, 0]} isAnimationActive={false}>
          <LabelList dataKey="gain" position="right" formatter={(v: number) => v.toFixed(0)}
            style={{ fontSize: 10, fill: "#1f2937" }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
