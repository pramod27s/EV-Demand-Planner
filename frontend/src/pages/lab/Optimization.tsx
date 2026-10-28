import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import ConvergenceChart from "../../components/charts/ConvergenceChart";
import ErrorBanner from "../../components/ui/ErrorBanner";
import HyperparamTable from "../../components/charts/HyperparamTable";
import Icon from "../../components/ui/Icon";
import { PageHeader, Section, Segmented } from "../../components/ui/Layout";
import Loader from "../../components/ui/Loader";
import { useApi } from "../../hooks/useApi";
import { COLORS, OPT_LABEL, OPT_ORDER } from "../../theme";
import type { OptimizerKey } from "../../types";

const STEP_MS = 300;
const SEED_DASH = [undefined, "6 3", "8 3 2 3", "2 3", "10 4"];

function Replay({ optimizer, seed }: { optimizer: OptimizerKey; seed: number }) {
  const { data, loading, error } = useApi(() => api.history(optimizer, seed), [optimizer, seed]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const n = data?.history.length ?? 0;

  // Show the full curve when a new run is loaded.
  useEffect(() => {
    setStep(n);
    setPlaying(false);
  }, [n, optimizer, seed]);

  useEffect(() => {
    if (!playing) return;
    if (step >= n) {
      setPlaying(false);
      return;
    }
    const id = setTimeout(() => setStep((s) => s + 1), STEP_MS);
    return () => clearTimeout(id);
  }, [playing, step, n]);

  if (loading) return <Loader rows={5} />;
  if (error || !data) return <ErrorBanner message={error ?? "No history"} />;

  const current = data.history[Math.max(0, step - 1)];
  const first = data.history[0].best_fitness;
  const gain = first - current.best_fitness;
  const color = COLORS[optimizer];
  const play = () => {
    setStep(step >= n ? 1 : step);
    setPlaying(true);
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div>
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <button onClick={playing ? () => setPlaying(false) : play} className="btn-primary px-5">
            <Icon name={playing ? "pause" : "play"} className="h-4 w-4" />{playing ? "Pause" : "Play"}
          </button>
          <button onClick={() => { setPlaying(false); setStep(n); }} className="btn-secondary">Show all</button>
          <div className="flex min-w-[160px] flex-1 items-center gap-3">
            <div className="h-2 flex-1 rounded-full bg-slate-100">
              <div className="h-2 rounded-full transition-all duration-200" style={{ width: `${(step / n) * 100}%`, background: color }} />
            </div>
            <span className="text-sm tabular-nums text-slate-500">{step}/{n}</span>
          </div>
        </div>
        <ConvergenceChart upTo={step} height={330} series={[
          { key: "best", label: "best so far", color, values: data.history.map((h) => h.best_fitness) },
          { key: "mean", label: "population mean", color: "#94a3b8", dash: "4 3", values: data.history.map((h) => h.mean_fitness) },
        ]} />
      </div>
      <div className="rounded border border-zinc-200 bg-zinc-50 p-4">
        <div className="flex items-center justify-between text-sm text-slate-500">
          <span>Best validation RMSE</span>
          <span className="rounded-md bg-white px-2 py-0.5 text-xs font-medium ring-1 ring-slate-200">iteration {current.iteration}</span>
        </div>
        <div className="mt-2 flex items-baseline gap-2">
          <span className="num text-3xl font-semibold text-zinc-900">{current.best_fitness.toFixed(4)}</span>
          <span className="text-sm text-slate-500">kWh</span>
        </div>
        <div className="mt-1 text-sm">
          {gain > 0
            ? <span className="font-medium text-green-700">−{gain.toFixed(4)} kWh since iteration 1</span>
            : <span className="text-slate-500">starting point</span>}
        </div>
        <div className="mt-4 rounded border border-zinc-200 bg-white p-3">
          <HyperparamTable columns={[{ key: "cur", label: "current best", color, params: current.best_params }]} />
        </div>
      </div>
    </div>
  );
}

export default function Optimization() {
  const runs = useApi(api.optimizationRuns);
  const conv = useApi(() => api.convergence());
  const hp = useApi(api.hyperparams);
  const [optimizer, setOptimizer] = useState<OptimizerKey>("sma");
  const [seed, setSeed] = useState<number | null>(null);

  const seeds = useMemo(() => runs.data?.[optimizer] ?? [], [runs.data, optimizer]);
  useEffect(() => {
    if (seeds.length && (seed === null || !seeds.includes(seed))) setSeed(seeds[0]);
  }, [seeds, seed]);

  if (runs.loading) return <Loader />;
  if (runs.error || !runs.data) return <ErrorBanner message={runs.error ?? "No runs"} />;

  const available = OPT_ORDER.filter((k) => runs.data![k]);
  const entry = conv.data?.[optimizer];

  return (
    <div className="space-y-8">
      <PageHeader eyebrow="Optimisation replay" title="Watch the search, iteration by iteration">
        Each optimiser explores a 9-dimensional XGBoost hyperparameter space under the same budget.
        Press play to see the best validation RMSE fall as the population moves.
      </PageHeader>

      <div className="card flex flex-wrap items-center gap-x-8 gap-y-3 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm font-medium text-slate-500">Optimiser</span>
          <Segmented label="Optimiser" value={optimizer} onChange={setOptimizer}
            options={available.map((k) => ({ value: k, label: OPT_LABEL[k], color: COLORS[k] }))} />
        </div>
        {seed !== null && (
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm font-medium text-slate-500">Seed</span>
            <Segmented label="Seed" value={seed} onChange={setSeed} options={seeds.map((s) => ({ value: s, label: String(s) }))} />
          </div>
        )}
      </div>

      <Section title={`${OPT_LABEL[optimizer]} · seed ${seed ?? "—"}`} sub="Best-so-far and population-mean validation RMSE per iteration" icon="trend">
        {seed !== null && <Replay optimizer={optimizer} seed={seed} />}
      </Section>

      <Section title={`Every seed of ${OPT_LABEL[optimizer]}`} sub="Line style distinguishes seeds; tight curves mean the optimiser is reliable" icon="layers">
        {conv.loading && <Loader rows={4} />}
        {conv.error && <ErrorBanner message={conv.error} />}
        {entry && (
          <ConvergenceChart series={Object.entries(entry.seeds).map(([s, values], i) => ({
            key: `seed${s}`, label: `seed ${s}`, color: COLORS[optimizer], values,
            dash: SEED_DASH[i % SEED_DASH.length],
          }))} />
        )}
      </Section>

      <Section title="Best hyperparameters per optimiser" sub="Best seed of each optimiser, selected on validation RMSE (never on test)" icon="sparkles">
        {hp.loading && <Loader rows={6} />}
        {hp.error && <ErrorBanner message={hp.error} />}
        {hp.data && (
          <>
            <HyperparamTable columns={OPT_ORDER.filter((k) => hp.data![k]).map((k) => ({
              key: k, label: `${OPT_LABEL[k]} · seed ${hp.data![k]!.seed}`, color: COLORS[k], params: hp.data![k]!.params,
            }))} />
            <div className="mt-4 flex flex-wrap gap-2">
              {OPT_ORDER.filter((k) => hp.data![k]).map((k) => (
                <span key={k} className="flex items-center gap-2 rounded border border-zinc-200 px-2.5 py-1 text-xs text-zinc-600">
                  <span className="h-2 w-2 rounded-full" style={{ background: COLORS[k] }} />
                  {OPT_LABEL[k]} val RMSE <b className="tabular-nums text-slate-900">{hp.data![k]!.best_val_rmse.toFixed(4)}</b>
                </span>
              ))}
            </div>
          </>
        )}
      </Section>
    </div>
  );
}
