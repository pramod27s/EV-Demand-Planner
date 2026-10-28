import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_TICK, TOOLTIP_STYLE } from "../../theme";

interface Props {
  actual: number[];
  predicted: number[];
  color: string;
  height?: number;
}

export default function PredVsActualScatter({ actual, predicted, color, height = 320 }: Props) {
  const points = actual.map((a, i) => ({ actual: a, predicted: predicted[i] }));
  const max = Math.ceil(Math.max(...actual, ...predicted));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 16, left: 0 }}>
        <CartesianGrid stroke="#eef2f7" />
        <XAxis type="number" dataKey="actual" name="Actual" domain={[0, max]}
          tick={AXIS_TICK} tickLine={false}
          label={{ value: "Actual (kWh)", position: "insideBottom", offset: -8, fontSize: 12, fill: "#6b7280" }} />
        <YAxis type="number" dataKey="predicted" name="Predicted" domain={[0, max]} width={48}
          tick={AXIS_TICK} tickLine={false} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => v.toFixed(2)} cursor={{ strokeDasharray: "3 3" }} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke="#1f2937" strokeDasharray="4 4"
          label={{ value: "y = x", position: "insideTopLeft", fontSize: 11, fill: "#1f2937" }} />
        <Scatter data={points} fill={color} fillOpacity={0.3} isAnimationActive={false} shape="circle" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
