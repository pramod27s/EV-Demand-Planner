import L from "leaflet";
import { useEffect, useMemo } from "react";
import { CircleMarker, MapContainer, Marker, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { AreaLocation, AreaSite } from "../../types";

/** Green (low) -> amber -> red (high). */
function heat(v: number, lo: number, hi: number): string {
  const t = hi > lo ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : 0.5;
  const stops = [[22, 163, 74], [234, 179, 8], [220, 38, 38]];
  const seg = t < 0.5 ? 0 : 1;
  const local = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const c = stops[seg].map((a, i) => Math.round(a + (stops[seg + 1][i] - a) * local));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

const siteIcon = (rank: number) =>
  L.divIcon({
    className: "",
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    html: `<div style="width:26px;height:26px;border-radius:9999px;background:#18181b;color:#fff;display:flex;align-items:center;justify-content:center;font:600 12px 'IBM Plex Mono',monospace;border:2px solid #fff">${rank}</div>`,
  });

function FitBounds({ points, signal }: { points: [number, number][]; signal: number }) {
  const map = useMap();
  useEffect(() => {
    if (points.length) map.fitBounds(L.latLngBounds(points), { padding: [40, 40], maxZoom: 15 });
  }, [map, points, signal]);
  return null;
}

/** Leaflet only measures its container once; re-measure when the panel divider moves. */
function FollowContainerSize() {
  const map = useMap();
  useEffect(() => {
    const ro = new ResizeObserver(() => map.invalidateSize());
    ro.observe(map.getContainer());
    return () => ro.disconnect();
  }, [map]);
  return null;
}

interface Props {
  locations: AreaLocation[];
  sites: AreaSite[];
  selected?: string | null;
  onSelect?: (l: AreaLocation) => void;
  measured?: boolean;   // demand files: kWh are metered, not estimated
  fitSignal?: number;   // change this number to re-fit the view to all points
}

/** Full-size map: locations sized by activity and coloured by kWh/day, plus numbered suggested sites. */
export default function AreaMap({ locations, sites, selected, onSelect, measured = false, fitSignal = 0 }: Props) {
  const daily = locations.map((l) => l.daily_kwh);
  const lo = Math.min(...daily);
  const hi = Math.max(...daily);
  const maxSessions = Math.max(...locations.map((l) => l.sessions));
  // memoised so the map only re-fits when the data changes, not when a location is selected
  const points = useMemo<[number, number][]>(
    () => [...locations.map((l) => [l.lat, l.lon] as [number, number]), ...sites.map((s) => [s.lat, s.lon] as [number, number])],
    [locations, sites],
  );

  return (
    <div className="relative h-full w-full">
      <MapContainer center={points[0] ?? [0, 0]} zoom={13} scrollWheelZoom className="h-full w-full">
        <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <FitBounds points={points} signal={fitSignal} />
        <FollowContainerSize />
        {locations.map((l) => (
          <CircleMarker key={l.name} center={[l.lat, l.lon]} radius={4 + 12 * Math.sqrt(l.sessions / maxSessions)}
            pathOptions={{ color: l.name === selected ? "#18181b" : "#ffffff", weight: l.name === selected ? 3 : 1.5,
              fillColor: heat(l.daily_kwh, lo, hi), fillOpacity: 0.85 }}
            eventHandlers={{ click: () => onSelect?.(l) }}>
            <Tooltip>
              <div className="text-xs">
                <div className="font-semibold">{l.name.replace("PALO ALTO CA / ", "")}</div>
                <div className="num">{l.daily_kwh.toFixed(1)} kWh/day · {l.sessions.toLocaleString()} {measured ? "months" : "sessions"}</div>
              </div>
            </Tooltip>
          </CircleMarker>
        ))}
        {sites.map((s) => (
          <Marker key={s.rank} position={[s.lat, s.lon]} icon={siteIcon(s.rank)} zIndexOffset={1000}>
            <Tooltip direction="top" offset={[0, -14]}>
              <div className="text-xs">
                <div className="font-semibold">Suggested site {s.rank}</div>
                <div className="num">{(s.share * 100).toFixed(0)}% of demand · {s.mean_distance_km.toFixed(2)} km avg</div>
              </div>
            </Tooltip>
          </Marker>
        ))}
      </MapContainer>
      <div className="pointer-events-none absolute bottom-6 left-3 z-[1000] rounded border border-zinc-300 bg-white/95 px-2.5 py-2 text-2xs text-zinc-700">
        <div className="mb-1 font-medium">{measured ? "Measured" : "Estimated"} kWh / day</div>
        <div className="h-1.5 w-36" style={{ background: "linear-gradient(90deg, rgb(22,163,74), rgb(234,179,8), rgb(220,38,38))" }} />
        <div className="num mt-0.5 flex justify-between"><span>{lo.toFixed(1)}</span><span>{hi.toFixed(1)}</span></div>
        <div className="mt-1 flex items-center gap-1.5 text-zinc-500">
          <span className="num inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-zinc-900 text-[8px] font-semibold text-white">1</span>
          suggested site · size = {measured ? "months" : "sessions"}
        </div>
      </div>
    </div>
  );
}
