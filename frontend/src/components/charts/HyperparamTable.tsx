import type { HyperParams } from "../../types";

const ORDER = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
  "min_child_weight", "gamma", "reg_lambda", "reg_alpha"];

const show = (v: number | undefined) =>
  v === undefined ? "—" : Number.isInteger(v) ? String(v) : Math.abs(v) < 0.01 ? v.toExponential(2) : v.toFixed(4);

interface Column {
  key: string;
  label: string;
  color?: string;
  params: HyperParams;
}

/** Hyperparameters as rows, one column per optimiser (or a single column during replay). */
export default function HyperparamTable({ columns }: { columns: Column[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-slate-500">
            <th className="py-2 pr-4 font-medium">Hyperparameter</th>
            {columns.map((c) => (
              <th key={c.key} className="py-2 pr-4 text-right font-medium">
                {c.color && <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full" style={{ background: c.color }} />}
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ORDER.map((p) => (
            <tr key={p} className="border-b border-slate-100">
              <td className="py-1.5 pr-4 font-mono text-xs text-slate-700">{p}</td>
              {columns.map((c) => (
                <td key={c.key} className="py-1.5 pr-4 text-right tabular-nums text-slate-900">{show(c.params[p])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
