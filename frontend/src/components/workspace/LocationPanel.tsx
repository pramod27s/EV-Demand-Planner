import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, errorMessage } from "../../api/client";
import { AXIS_TICK, COLORS, TOOLTIP_STYLE } from "../../theme";
import type { AreaEstimate, AreaLocation } from "../../types";
import ErrorBanner from "../ui/ErrorBanner";
import { Stat } from "../ui/Layout";

// Typical charging power per port type, used only for the approximate charging-time hint.
const PORT_KW: Record<string, number> = { "Level 1": 1.4, "Level 2": 6.6 };
const shortName = (n: string) => n.replace("PALO ALTO CA / ", "");
const hours = (h: number) => (h < 1 ? `${Math.round(h * 60)} min` : `${Math.floor(h)} h ${Math.round((h % 1) * 60)} min`);

function Title({ children }: { children: React.ReactNode }) {
  return <h3 className="panel-title mb-1">{children}</h3>;
}

/** Details of one location plus an "estimate a session here" form (session files only). */
export default function LocationPanel({ loc, onBack, demand = false }: { loc: AreaLocation; onBack: () => void; demand?: boolean }) {
  return (
    <div>
      <div className="sticky top-0 z-10 flex items-start justify-between gap-2 border-b border-zinc-200 bg-white px-4 py-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold text-zinc-900">{shortName(loc.name)}</h2>
          <div className="num text-2xs text-zinc-500">
            {loc.lat.toFixed(5)}, {loc.lon.toFixed(5)}{loc.port_type ? ` · ${loc.port_type}` : ""}{loc.plug_type ? ` · ${loc.plug_type}` : ""}
          </div>
        </div>
        <button onClick={onBack} className="rounded px-1.5 text-lg leading-none text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900" aria-label="Close">×</button>
      </div>
      <div className="space-y-4 px-4 py-3">{demand ? <DemandDetails loc={loc} /> : <SessionDetails loc={loc} />}</div>
    </div>
  );
}

function SessionDetails({ loc }: { loc: AreaLocation }) {
  const [form, setForm] = useState({
    date: new Date().toISOString().slice(0, 10), time: "09:00", port_type: loc.port_type || "Level 2", sessions: 0, mean: "",
  });
  const [est, setEst] = useState<AreaEstimate | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const avg = loc.mean_actual_kwh ?? loc.mean_kwh;
  const hourly = loc.hourly_mean_kwh.map((v, h) => ({ hour: h, kwh: v }));
  const kw = PORT_KW[form.port_type] ?? PORT_KW["Level 2"];

  async function estimate(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setEst(await api.estimateAt({
        name: loc.name, lat: loc.lat, lon: loc.lon, location_mean_kwh: avg,
        port_type: form.port_type, plug_type: loc.plug_type || "J1772", port_number: 1,
        start_datetime: `${form.date}T${form.time}:00`, user_prev_sessions: Number(form.sessions),
        user_prev_mean_kwh: form.mean === "" ? null : Number(form.mean),
      }));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="divide-y divide-zinc-100">
        <Stat label="Sessions" value={loc.sessions.toLocaleString()} />
        <Stat label="Energy per day" value={`${loc.daily_kwh.toFixed(1)} kWh`} />
        <Stat label="Average session" value={`${avg.toFixed(1)} kWh`} />
        <Stat label="Busiest hour" value={loc.peak_hour != null ? `${loc.peak_hour}:00` : "—"} />
      </div>
      <div>
        <Title>Average session kWh by plug-in hour</Title>
        <ResponsiveContainer width="100%" height={110}>
          <BarChart data={hourly} margin={{ top: 4, right: 0, bottom: 0, left: -26 }}>
            <CartesianGrid stroke="#f0f0f1" vertical={false} />
            <XAxis dataKey="hour" tick={AXIS_TICK} interval={5} tickLine={false} />
            <YAxis tick={AXIS_TICK} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(1)} kWh`, "average"]} labelFormatter={(h) => `${h}:00`} />
            <Bar dataKey="kwh" fill={COLORS.sma} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <form onSubmit={estimate} className="space-y-2 border-t border-zinc-200 pt-3">
        <Title>Estimate a session here</Title>
        <div className="grid grid-cols-2 gap-2">
          <label className="label">Date
            <input type="date" className="input" value={form.date} onChange={(e) => setForm({ ...form, date: e.target.value })} required />
          </label>
          <label className="label">Plug-in time
            <input type="time" className="input" value={form.time} onChange={(e) => setForm({ ...form, time: e.target.value })} required />
          </label>
          <label className="label">Port
            <select className="input" value={form.port_type} onChange={(e) => setForm({ ...form, port_type: e.target.value })}>
              {Object.keys(PORT_KW).map((p) => <option key={p}>{p}</option>)}
            </select>
          </label>
          <label className="label">Charged here before
            <input type="number" min={0} className="input" value={form.sessions} onChange={(e) => setForm({ ...form, sessions: Number(e.target.value) })} />
          </label>
        </div>
        <label className="label">Usual kWh per session (optional)
          <input type="number" min={0} max={100} step="0.1" className="input" placeholder="leave blank if unsure"
            value={form.mean} onChange={(e) => setForm({ ...form, mean: e.target.value })} />
        </label>
        <button type="submit" disabled={busy} className="btn-primary w-full">{busy ? "Estimating…" : "Estimate"}</button>
        {error && <ErrorBanner message={error} />}
        {est && (
          <div className="rounded border border-zinc-300 bg-zinc-50 px-3 py-2.5">
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-zinc-500">Estimated energy</span>
              <span className="num text-2xs text-zinc-500">likely {est.low_kwh.toFixed(1)}–{est.high_kwh.toFixed(1)}</span>
            </div>
            <div className="num text-3xl font-semibold text-zinc-900">{est.kwh.toFixed(1)} <span className="text-sm font-normal text-zinc-500">kWh</span></div>
            <div className="mt-1 divide-y divide-zinc-200 border-t border-zinc-200">
              <Stat label="Charging time" value={`≈ ${hours(est.kwh / kw)}`} sub={`at ${kw} kW`} />
              <Stat label="vs this location" value={Math.abs(est.kwh - avg) / avg < 0.03 ? "average"
                : `${Math.abs(((est.kwh - avg) / avg) * 100).toFixed(0)}% ${est.kwh > avg ? "above" : "below"}`} />
            </div>
          </div>
        )}
      </form>
    </>
  );
}

/** Measured monthly demand of one locality (demand files have no sessions to estimate). */
function DemandDetails({ loc }: { loc: AreaLocation }) {
  const months = loc.monthly_kwh ?? [];
  const peak = months.reduce((a, b) => (b.kwh > (a?.kwh ?? -1) ? b : a), months[0]);
  return (
    <>
      <div className="divide-y divide-zinc-100">
        <Stat label="Months of data" value={String(loc.sessions)} />
        <Stat label="Average per month" value={`${Math.round(loc.mean_kwh).toLocaleString()} kWh`} />
        <Stat label="Energy per day" value={`${loc.daily_kwh.toFixed(1)} kWh`} />
        <Stat label="Busiest month" value={peak ? peak.month : "—"} />
      </div>
      <div>
        <Title>Measured kWh by month</Title>
        <ResponsiveContainer width="100%" height={130}>
          <BarChart data={months} margin={{ top: 4, right: 0, bottom: 0, left: -16 }}>
            <CartesianGrid stroke="#f0f0f1" vertical={false} />
            <XAxis dataKey="month" tick={AXIS_TICK} tickLine={false} interval={Math.max(0, Math.floor(months.length / 4))} />
            <YAxis tick={AXIS_TICK} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${Math.round(v).toLocaleString()} kWh`, "energy"]} />
            <Bar dataKey="kwh" fill={COLORS.sma} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-2xs leading-relaxed text-zinc-500">
        Metered kWh for this locality — nothing to estimate. Files with individual sessions (a start time per row) get per-session estimates.
      </p>
    </>
  );
}
