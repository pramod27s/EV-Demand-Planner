import { useState } from "react";
import { api } from "../../api/client";
import ComparisonBarChart from "../../components/charts/ComparisonBarChart";
import ConvergenceChart from "../../components/charts/ConvergenceChart";
import ErrorBanner from "../../components/ui/ErrorBanner";
import FeatureImportanceChart from "../../components/charts/FeatureImportanceChart";
import { Badge, Section, Segmented } from "../../components/ui/Layout";
import Loader from "../../components/ui/Loader";
import MetricCard from "../../components/ui/MetricCard";
import PredVsActualScatter from "../../components/charts/PredVsActualScatter";
import { useApi } from "../../hooks/useApi";
import { COLORS, DASH, MODEL_COLOR, OPT_LABEL, OPT_ORDER, fmt } from "../../theme";
import type { ComparisonRow } from "../../types";

/** Ranked list of every model by test RMSE, with a bar showing distance from the best. */
function Leaderboard({ rows }: { rows: ComparisonRow[] }) {
  const ranked = [...rows].sort((a, b) => a.rmse - b.rmse);
  const best = ranked[0].rmse;
  const worst = ranked[ranked.length - 1].rmse;
  return (
    <ol className="space-y-2.5">
      {ranked.map((r, i) => {
        const width = worst > best ? 30 + 70 * (1 - (r.rmse - best) / (worst - best)) : 100;
        const isSma = r.model === "XGBoost + SMA";
        return (
          <li key={r.model} className={`rounded px-2 py-1.5 ${isSma ? "bg-zinc-100" : ""}`}>
            <div className="flex items-center gap-3 text-sm">
              <span className="w-5 text-right font-semibold tabular-nums text-slate-400">{i + 1}</span>
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: MODEL_COLOR[r.model] }} />
              <span className={`flex-1 truncate ${isSma ? "font-semibold text-slate-900" : "text-slate-700"}`}>{r.model}</span>
              <span className="font-semibold tabular-nums text-slate-900">{r.rmse.toFixed(3)}</span>
              <span className={`w-16 text-right text-xs font-medium tabular-nums ${r.rmse_improvement_pct > 0 ? "text-green-700" : r.rmse_improvement_pct < 0 ? "text-red-600" : "text-slate-400"}`}>
                {r.rmse_improvement_pct === 0 ? "baseline" : `${r.rmse_improvement_pct > 0 ? "−" : "+"}${Math.abs(r.rmse_improvement_pct).toFixed(2)}%`}
              </span>
            </div>
            <div className="ml-8 mt-1.5 h-1.5 rounded-full bg-slate-100">
              <div className="h-1.5 rounded-full" style={{ width: `${width}%`, background: MODEL_COLOR[r.model] }} />
            </div>
          </li>
        );
      })}
    </ol>
  );
}

export default function Performance() {
  const comp = useApi(api.comparison);
  const conv = useApi(() => api.convergence());
  const ds = useApi(api.datasetSummary);
  const fi = useApi(api.featureImportance);
  const [scatterModel, setScatterModel] = useState<"sma" | "default">("sma");
  const sample = useApi(() => api.testSample(scatterModel), [scatterModel]);

  if (comp.loading) return <Loader />;
  if (comp.error || !comp.data) return <ErrorBanner message={comp.error ?? "No data"} />;

  const rows = comp.data.rows;
  const dflt = rows.find((r) => r.model === "XGBoost (default)");
  const sma = rows.find((r) => r.model === "XGBoost + SMA");
  const improvement = sma?.rmse_improvement_pct ?? 0;

  return (
    <div className="space-y-8">
      {/* ---------- header ---------- */}
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-zinc-200 pb-4">
        <div>
          <h1 className="text-xl font-semibold text-zinc-900">Model performance</h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-600">
            XGBoost predicts session energy from plug-in-time information. The Slime Mould Algorithm tuned its 9 hyperparameters;
            all numbers are on the held-out test set (35,528 sessions, Apr–Dec 2019).
            {comp.data.quick && <span className="ml-1 font-medium text-amber-700">(Quick smoke-test results.)</span>}
          </p>
        </div>
        <div className="num text-right text-sm">
          <div className="text-zinc-500">test RMSE, kWh</div>
          <div><span className="text-zinc-500">default</span> {fmt(dflt?.rmse)} → <span className="font-semibold text-zinc-900">SMA {fmt(sma?.rmse)}</span>
            <span className="ml-2 text-green-700">−{improvement.toFixed(2)}%</span></div>
        </div>
      </div>

      {/* ---------- KPIs ---------- */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Default XGBoost" value={fmt(dflt?.rmse)} unit="kWh RMSE" icon="target" accent={COLORS.default} sub="no optimisation" />
        <MetricCard label="SMA-tuned XGBoost" value={fmt(sma?.rmse)} unit="kWh RMSE" icon="sparkles" accent={COLORS.sma}
          delta={{ text: `−${improvement.toFixed(2)}%`, good: improvement > 0 }} sub="vs default" />
        <MetricCard label="Explained variance" value={fmt(sma?.r2)} unit="R²" icon="trend" accent="#0d9488"
          delta={dflt && sma ? { text: `+${(sma.r2 - dflt.r2).toFixed(3)}`, good: sma.r2 > dflt.r2 } : undefined} sub="SMA model" />
        <MetricCard label="Charging sessions" value={ds.data ? ds.data.n_sessions.toLocaleString() : "—"} icon="database" accent="#64748b"
          sub={ds.data ? `${ds.data.n_stations} stations · 2012–2019` : undefined} />
      </div>

      {/* ---------- leaderboard + convergence ---------- */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Section title="Model leaderboard" sub="Ranked by test RMSE · change vs default XGBoost" icon="chart">
          <Leaderboard rows={rows} />
        </Section>
        <Section title="How fast each optimiser converges" icon="trend"
          sub="Best validation RMSE so far, mean over seeds · same budget for all"
          actions={<Badge tone="slate">{conv.data?.sma ? `${Object.keys(conv.data.sma.seeds).length} seed(s) each` : "…"}</Badge>}>
          {conv.loading && <Loader rows={4} />}
          {conv.error && <ErrorBanner message={conv.error} />}
          {conv.data && (
            <ConvergenceChart height={360} series={OPT_ORDER.filter((k) => conv.data![k]).map((k) => ({
              key: k, label: OPT_LABEL[k], color: COLORS[k], dash: DASH[k], values: conv.data![k]!.mean,
            }))} />
          )}
        </Section>
      </div>

      {/* ---------- metric bars ---------- */}
      <Section title="Baselines vs optimised models" sub="Test-set RMSE, MAE and R² · y-axes zoomed to show differences" icon="layers">
        <ComparisonBarChart rows={rows} />
      </Section>

      {/* ---------- scatter + importances ---------- */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Section title="Predicted vs actual" sub="2,000 random test sessions" icon="target"
          actions={<Segmented label="Model" value={scatterModel} onChange={setScatterModel}
            options={[{ value: "sma", label: "SMA", color: COLORS.sma }, { value: "default", label: "Default", color: COLORS.default }]} />}>
          {sample.loading && <Loader rows={4} />}
          {sample.error && <ErrorBanner message={sample.error} />}
          {sample.data && <PredVsActualScatter actual={sample.data.actual} predicted={sample.data.predicted}
            color={scatterModel === "sma" ? COLORS.sma : COLORS.default} />}
        </Section>
        <Section title="What drives the prediction" sub="Top-15 feature importances (gain) of the SMA model" icon="sparkles">
          {fi.loading && <Loader rows={6} />}
          {fi.error && <ErrorBanner message={fi.error} />}
          {fi.data && <FeatureImportanceChart items={fi.data} />}
        </Section>
      </div>
    </div>
  );
}
