import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, errorMessage } from "../api/client";
import ErrorBanner from "../components/ui/ErrorBanner";
import { Segmented, Stat } from "../components/ui/Layout";
import AreaMap from "../components/workspace/AreaMap";
import LocationPanel from "../components/workspace/LocationPanel";
import { AXIS_TICK, COLORS, TOOLTIP_STYLE } from "../theme";
import type { AreaAnalysis } from "../types";

type Mode = "samples" | "upload";

const SAMPLE_ID = "palo_alto";
// Resizable layout: side-panel width and chart-strip height (px), remembered per browser.
interface Size { key: string; def: number; min: number; max: number }
const PANEL: Size = { key: "workspace.panelWidth", def: 380, min: 300, max: 640 };
const CHART: Size = { key: "workspace.chartHeight", def: 170, min: 110, max: 460 };
const MIN_MAP_HEIGHT = 200;   // the chart strip may never squeeze the map below this
const clamp = (sz: Size, v: number) => Math.min(sz.max, Math.max(sz.min, Math.round(v)));

function readSize(sz: Size): number {
  try {
    const v = Number(localStorage.getItem(sz.key));
    return v ? clamp(sz, v) : sz.def;
  } catch {
    return sz.def;
  }
}

function saveSize(sz: Size, v: number) {
  try {
    localStorage.setItem(sz.key, String(v));
  } catch {
    /* storage unavailable (private mode): keep the size for this visit only */
  }
}

/** Pointer-drag helper: `toValue` maps the pointer position to a size; saved on release. */
function startDrag(e: React.PointerEvent, cursor: string, toValue: (ev: PointerEvent) => number,
                   set: (v: number) => void, save: (v: number) => void) {
  e.preventDefault();
  let last: number | null = null;
  const move = (ev: PointerEvent) => {
    last = toValue(ev);
    set(last);
  };
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    if (last !== null) save(last);
  };
  document.body.style.userSelect = "none";
  document.body.style.cursor = cursor;
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
}
const DAYS_PER_MONTH = 30.44;
const num = (v: number, d = 0) => v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const shortName = (n: string) => n.replace("PALO ALTO CA / ", "");

function download(text: string, name: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

/** One block of the side panel. */
function Block({ title, action, children }: { title: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="border-b border-zinc-200 px-4 py-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h2 className="panel-title">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

/** The tool: pick a dataset, read the demand map, click locations, get SMA charger sites. */
export default function Workspace() {
  const [params] = useSearchParams();
  const inputRef = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<Mode>(params.get("mode") === "upload" ? "upload" : "samples");
  const [file, setFile] = useState<{ name: string; body: Blob | string } | null>(null);
  const [k, setK] = useState(3);
  const [result, setResult] = useState<AreaAnalysis | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  // Every load gets a ticket; a response whose ticket is no longer current (the user switched
  // dataset or started a newer load meanwhile) is ignored instead of overwriting the screen.
  const ticket = useRef(0);
  const [panelW, setPanelW] = useState(() => readSize(PANEL));
  const [chartH, setChartH] = useState(() => readSize(CHART));
  const [maximised, setMaximised] = useState(false);   // map fills the window
  const [fitSignal, setFitSignal] = useState(0);       // bump to re-fit the map to the data
  const asideRef = useRef<HTMLElement>(null);
  const mainRef = useRef<HTMLElement>(null);

  const setPanel = (v: number) => { const n = clamp(PANEL, v); setPanelW(n); saveSize(PANEL, n); };
  const setChart = (v: number) => { const n = clamp(CHART, v); setChartH(n); saveSize(CHART, n); };

  function resizePanel(e: React.PointerEvent) {
    const left = asideRef.current?.getBoundingClientRect().left ?? 0;
    startDrag(e, "col-resize", (ev) => clamp(PANEL, ev.clientX - left), setPanelW, (v) => saveSize(PANEL, v));
  }

  function resizeChart(e: React.PointerEvent) {
    const box = mainRef.current?.getBoundingClientRect();
    const bottom = box?.bottom ?? window.innerHeight;
    const roomForChart = (box?.height ?? window.innerHeight) - MIN_MAP_HEIGHT;
    startDrag(e, "row-resize", (ev) => Math.min(clamp(CHART, bottom - ev.clientY), Math.max(CHART.min, roomForChart)),
              setChartH, (v) => saveSize(CHART, v));
  }

  // Esc leaves the full-map view
  useEffect(() => {
    if (!maximised) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setMaximised(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [maximised]);

  async function run(body: Blob | string, sites: number, preselect: string | null = null, id = ++ticket.current) {
    setBusy(true);
    setError(null);
    try {
      const r = await api.analyzeArea(body, sites);
      if (id !== ticket.current) return;              // stale response
      setResult(r);
      setSelected(preselect && r.locations.some((l) => l.name === preselect) ? preselect : null);
    } catch (e) {
      if (id !== ticket.current) return;
      setError(errorMessage(e));
      setResult(null);
    } finally {
      if (id === ticket.current) setBusy(false);
    }
  }

  async function loadSample(preselect: string | null = null) {
    const id = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const csv = await api.areaSampleCsv(SAMPLE_ID);
      if (id !== ticket.current) return;              // user moved on while downloading
      setFile({ name: "Palo Alto sample", body: csv });
      await run(csv, k, preselect, id);
    } catch (e) {
      if (id !== ticket.current) return;
      setError(errorMessage(e));
      setBusy(false);
    }
  }

  function chooseFile(f: File | undefined) {
    if (!f) return;
    setFile({ name: f.name, body: f });
    run(f, k);
  }

  function switchMode(m: Mode) {
    ticket.current++;                                 // cancel any load still in flight
    setBusy(false);
    setMode(m);
    setResult(null);
    setSelected(null);
    setFile(null);
    setError(null);
    if (m === "samples") loadSample();
  }

  // The sample loads on arrival; ?loc=<name> pre-selects a location.
  useEffect(() => {
    if (mode === "samples") loadSample(params.get("loc"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const changeK = (v: number) => {
    const next = Math.min(10, Math.max(1, v));
    setK(next);
    if (file) run(file.body, next, selected);
  };

  async function downloadEstimates() {
    if (!file) return;
    try {
      download(await api.areaPredictionsCsv(file.body), "energy_estimates.csv");
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  const s = result?.summary;
  const demand = s?.mode === "demand";
  const selectedLoc = result?.locations.find((l) => l.name === selected) ?? null;

  return (
    <div className="flex h-full flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
      {/* ================= side panel ================= */}
      <aside ref={asideRef} style={{ "--panel-w": `${panelW}px` } as React.CSSProperties}
        className={`w-full shrink-0 overflow-y-auto bg-white lg:h-full lg:w-[var(--panel-w)] ${maximised ? "hidden" : ""}`}>
        <Block title="Data">
          <Segmented label="Dataset" value={mode} onChange={switchMode}
            options={[{ value: "samples", label: "Palo Alto sample" }, { value: "upload", label: "Upload CSV" }]} />
          {mode === "samples" ? (
            <p className="mt-2 text-xs leading-relaxed text-zinc-500">
              8,000 recent sessions at the City of Palo Alto's public stations — the network the model was trained on.
            </p>
          ) : (
            <div className="mt-2 space-y-2">
              <div
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files[0]); }}
                className={`rounded border border-dashed px-3 py-3 text-center ${dragging ? "border-zinc-900 bg-zinc-50" : "border-zinc-300"}`}>
                <div className="truncate text-sm font-medium text-zinc-800">{file ? file.name : "Drop a CSV file here"}</div>
                <button className="btn-secondary mt-2 px-2.5 py-1 text-xs" onClick={() => inputRef.current?.click()} disabled={busy}>
                  {file ? "Choose another file" : "Choose file"}
                </button>
                <input ref={inputRef} type="file" accept=".csv,text/csv" className="hidden"
                  onChange={(e) => { chooseFile(e.target.files?.[0]); e.target.value = ""; }} />
              </div>
              <p className="text-2xs leading-relaxed text-zinc-500">
                <b className="font-medium text-zinc-700">Sessions:</b> start time, latitude, longitude (optional: station, port, plug,
                driver ID, kWh). <b className="font-medium text-zinc-700">Demand:</b> location, latitude, longitude, kWh (optional: month).
                Up to 25 MB. <a href={api.areaTemplateUrl} className="font-medium text-zinc-900 underline">Template</a>
              </p>
            </div>
          )}
        </Block>

        <Block title="New charger sites">
          <div className="flex items-center gap-2">
            <button className="btn-secondary h-7 w-7 px-0" onClick={() => changeK(k - 1)} disabled={k <= 1 || busy} aria-label="Fewer sites">−</button>
            <span className="num w-6 text-center text-sm font-semibold">{k}</span>
            <button className="btn-secondary h-7 w-7 px-0" onClick={() => changeK(k + 1)} disabled={k >= 10 || busy} aria-label="More sites">+</button>
            <span className="text-xs text-zinc-500">placed by the Slime Mould Algorithm</span>
          </div>
        </Block>

        {error && <div className="px-4 pt-3"><ErrorBanner message={error} /></div>}

        {result && s && (
          <>
            <Block title="Summary">
              {s.far_from_training && (
                <p className="mb-2 rounded border border-amber-300 bg-amber-50 px-2.5 py-1.5 text-2xs leading-relaxed text-amber-900">
                  {num(s.distance_from_training_km)} km from Palo Alto, where the model was trained: compare locations with each other;
                  exact kWh may not transfer.
                </p>
              )}
              <div className="divide-y divide-zinc-100">
                <Stat label={demand ? "Records" : "Sessions"} value={num(s.rows_used)}
                  sub={s.rows_dropped ? `${num(s.rows_dropped)} rows skipped` : undefined} />
                <Stat label={demand ? "Localities" : "Locations"} value={num(s.n_locations)} />
                <Stat label="Period" value={demand ? `${s.date_start} → ${s.date_end}` : `${s.date_start.slice(0, 10)} → ${s.date_end.slice(0, 10)}`} />
                <Stat label="Energy demand" value={`${num(s.daily_kwh, 1)} kWh/day`}
                  sub={demand ? `${num(s.daily_kwh * DAYS_PER_MONTH)} kWh/month` : `avg session ${s.mean_kwh.toFixed(1)} kWh`} />
                {demand ? (
                  <Stat label="Values" value="measured" sub="metered kWh, no model needed" />
                ) : (
                  <Stat label="Estimate error" value={s.accuracy ? `± ${s.accuracy.rmse.toFixed(1)} kWh` : "—"}
                    sub={s.accuracy ? "RMSE on this file" : "add an energy column to check"} />
                )}
              </div>
            </Block>

            <Block title={`Suggested sites (${result.placement.k})`}>
              <ol className="space-y-2">
                {result.placement.sites.map((site) => (
                  <li key={site.rank} className="flex gap-2.5">
                    <span className="num mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-zinc-900 text-2xs font-semibold text-white">
                      {site.rank}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-sm font-medium">{(site.share * 100).toFixed(0)}% of demand</span>
                        <a className="text-2xs text-zinc-500 underline hover:text-zinc-900" target="_blank" rel="noreferrer"
                          href={`https://www.openstreetmap.org/?mlat=${site.lat}&mlon=${site.lon}#map=17/${site.lat}/${site.lon}`}>map ↗</a>
                      </div>
                      <div className="mt-1 h-1 rounded-sm bg-zinc-100"><div className="h-1 rounded-sm bg-green-600" style={{ width: `${site.share * 100}%` }} /></div>
                      <div className="num mt-1 truncate text-2xs text-zinc-500" title="location · locations served · average distance">
                        {site.lat.toFixed(4)}, {site.lon.toFixed(4)} · {site.served_locations} loc · {site.mean_distance_km.toFixed(2)} km
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
              <p className="mt-2 text-2xs text-zinc-500">
                Demand is on average <span className="num font-medium text-zinc-800">{result.placement.objective_km.toFixed(2)} km</span> from
                its nearest suggested site.
              </p>
            </Block>

            <Block title={`Locations (${result.locations.length})`}
              action={demand ? undefined : <button className="text-2xs font-medium text-zinc-700 underline hover:text-zinc-900" onClick={downloadEstimates}>Download estimates</button>}>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-2xs text-zinc-500">
                    <th className="pb-1 font-medium">Name</th>
                    <th className="pb-1 text-right font-medium">kWh/day</th>
                    <th className="pb-1 text-right font-medium">{demand ? "Months" : "Peak"}</th>
                  </tr>
                </thead>
                <tbody>
                  {result.locations.slice(0, 200).map((l) => (
                    <tr key={l.name} onClick={() => setSelected(l.name)}
                      className={`cursor-pointer border-t border-zinc-100 hover:bg-zinc-50 ${l.name === selected ? "bg-zinc-100 font-medium" : ""}`}>
                      <td className="max-w-[170px] truncate py-1 pr-2">{shortName(l.name)}</td>
                      <td className="num py-1 text-right">{l.daily_kwh.toFixed(1)}</td>
                      <td className="num py-1 text-right text-zinc-500">{demand ? l.sessions : l.peak_hour != null ? `${l.peak_hour}:00` : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Block>

            <Block title="How sites are chosen">
              <p className="text-2xs leading-relaxed text-zinc-500">
                Each location is weighted by its energy. The Slime Mould Algorithm tried {num(result.placement.evaluations)} placements of{" "}
                {result.placement.k} sites and kept the one with the shortest average distance from demand to a charger.
              </p>
              <ResponsiveContainer width="100%" height={70}>
                <LineChart data={result.placement.convergence_km.map((v, i) => ({ i: i + 1, v }))} margin={{ top: 6, right: 4, bottom: 0, left: 4 }}>
                  <YAxis hide domain={["dataMin", "dataMax"]} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(3)} km`, "avg distance"]} labelFormatter={(i) => `step ${i}`} />
                  <Line dataKey="v" stroke={COLORS.sma} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </Block>
          </>
        )}
      </aside>

      {/* drag to resize the side panel (double-click resets, arrow keys adjust) */}
      <div role="separator" aria-orientation="vertical" aria-label="Resize side panel" tabIndex={0}
        aria-valuenow={panelW} aria-valuemin={PANEL.min} aria-valuemax={PANEL.max}
        onPointerDown={resizePanel} onDoubleClick={() => setPanel(PANEL.def)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setPanel(panelW - 20);
          if (e.key === "ArrowRight") setPanel(panelW + 20);
        }}
        title="Drag to resize · double-click to reset"
        className={`group relative hidden w-1 shrink-0 cursor-col-resize bg-zinc-200 hover:bg-zinc-400 focus:bg-zinc-500 focus:outline-none ${maximised ? "" : "lg:block"}`}>
        <span className="absolute inset-y-0 -left-1.5 -right-1.5" />
        <span className="absolute left-1/2 top-1/2 h-8 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-zinc-400 group-hover:bg-zinc-600" />
      </div>

      {/* ================= map + chart strip ================= */}
      <main ref={mainRef} className="flex min-h-[560px] flex-1 flex-col lg:min-h-0">
        <div className="relative min-h-[380px] flex-1 lg:min-h-[200px]">
          {result ? (
            <AreaMap locations={result.locations} sites={result.placement.sites} selected={selected}
              onSelect={(l) => setSelected(l.name)} measured={demand} fitSignal={fitSignal} />
          ) : (
            <div className="flex h-full items-center justify-center bg-zinc-100 text-sm text-zinc-500">
              {busy ? "Loading…" : mode === "upload" ? "Upload a CSV file to see its locations on the map." : ""}
            </div>
          )}
          {result && (
            <div className="absolute left-[10px] top-[86px] z-[1000] flex flex-col overflow-hidden rounded border-2 border-black/20 bg-white">
              <button onClick={() => setFitSignal((n) => n + 1)} title="Fit all locations in view"
                className="flex h-[30px] w-[30px] items-center justify-center border-b border-zinc-300 text-base hover:bg-zinc-100">⤢</button>
              <button onClick={() => setMaximised((m) => !m)} title={maximised ? "Show panels (Esc)" : "Full map"}
                aria-pressed={maximised}
                className={`flex h-[30px] w-[30px] items-center justify-center text-sm hover:bg-zinc-100 ${maximised ? "bg-zinc-900 text-white hover:bg-zinc-700" : ""}`}>
                {maximised ? "✕" : "⛶"}
              </button>
            </div>
          )}
          {maximised && (
            <div className="pointer-events-none absolute right-3 top-3 z-[1000] rounded border border-zinc-300 bg-white/95 px-2.5 py-1 text-2xs text-zinc-600">
              Full map · Esc to show panels
            </div>
          )}
          {busy && result && (
            <div className="absolute left-1/2 top-3 z-[1000] -translate-x-1/2 rounded border border-zinc-300 bg-white px-3 py-1 text-xs text-zinc-700">
              Updating…
            </div>
          )}
          {selectedLoc && (
            <div className="absolute inset-y-3 right-3 z-[1000] w-[340px] max-w-[calc(100%-1.5rem)] overflow-y-auto rounded-md border border-zinc-300 bg-white shadow-md">
              <LocationPanel key={selectedLoc.name} loc={selectedLoc} demand={demand} onBack={() => setSelected(null)} />
            </div>
          )}
        </div>

        {result && s && !maximised && (
          <div role="separator" aria-orientation="horizontal" aria-label="Resize chart strip" tabIndex={0}
            aria-valuenow={chartH} aria-valuemin={CHART.min} aria-valuemax={CHART.max}
            onPointerDown={resizeChart} onDoubleClick={() => setChart(CHART.def)}
            onKeyDown={(e) => {
              if (e.key === "ArrowUp") setChart(chartH + 20);
              if (e.key === "ArrowDown") setChart(chartH - 20);
            }}
            title="Drag to resize · double-click to reset"
            className="group relative h-1 shrink-0 cursor-row-resize bg-zinc-200 hover:bg-zinc-400 focus:bg-zinc-500 focus:outline-none">
            <span className="absolute -bottom-1.5 -top-1.5 inset-x-0" />
            <span className="absolute left-1/2 top-1/2 h-1 w-8 -translate-x-1/2 -translate-y-1/2 rounded-full bg-zinc-400 group-hover:bg-zinc-600" />
          </div>
        )}
        {result && s && !maximised && (
          <div style={{ height: chartH, maxHeight: `calc(100% - ${MIN_MAP_HEIGHT}px)` }}
            className="flex shrink-0 flex-col bg-white px-4 pb-2 pt-2">
            <div className="flex items-baseline justify-between">
              <h2 className="panel-title">{demand ? "Measured energy by month" : "Energy by plug-in hour (kWh/day)"}</h2>
              <span className="text-2xs text-zinc-500">all locations</span>
            </div>
            <div className="min-h-0 flex-1">
            <ResponsiveContainer width="100%" height="100%">
              {demand ? (
                <BarChart data={result.monthly} margin={{ top: 6, right: 4, bottom: 0, left: -8 }}>
                  <CartesianGrid stroke="#f0f0f1" vertical={false} />
                  <XAxis dataKey="month" tick={AXIS_TICK} tickLine={false} interval={Math.max(0, Math.floor(result.monthly.length / 10))} />
                  <YAxis tick={AXIS_TICK} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${Math.round(v / 1000)}k`} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${num(v)} kWh`, "energy"]} cursor={{ fill: "#f4f4f5" }} />
                  <Bar dataKey="total_kwh" fill={COLORS.sma} isAnimationActive={false} />
                </BarChart>
              ) : (
                <BarChart data={result.hourly.map((h) => ({ ...h, daily: h.total_kwh / s.days }))} margin={{ top: 6, right: 4, bottom: 0, left: -8 }}>
                  <CartesianGrid stroke="#f0f0f1" vertical={false} />
                  <XAxis dataKey="hour" tick={AXIS_TICK} tickLine={false} interval={1} tickFormatter={(h) => `${h}`} />
                  <YAxis tick={AXIS_TICK} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(1)} kWh/day`, "energy"]} labelFormatter={(h) => `${h}:00`} cursor={{ fill: "#f4f4f5" }} />
                  <Bar dataKey="daily" fill={COLORS.sma} isAnimationActive={false} />
                </BarChart>
              )}
            </ResponsiveContainer>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
