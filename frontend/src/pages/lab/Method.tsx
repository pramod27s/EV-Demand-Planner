import Icon, { type IconName } from "../../components/ui/Icon";
import { PageHeader, Section } from "../../components/ui/Layout";

function SmaLoopDiagram() {
  // Simple SVG of one SMA iteration. Uses currentColor-friendly neutrals plus SMA green.
  const box = "fill-white stroke-slate-300";
  return (
    <svg viewBox="-24 0 664 300" className="h-auto w-full max-w-2xl" role="img" aria-label="Diagram of the SMA loop">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" className="fill-slate-500" />
        </marker>
      </defs>
      <g fontSize="13" className="fill-slate-800">
        <rect x="20" y="20" width="170" height="54" rx="10" className={box} />
        <text x="105" y="43" textAnchor="middle" fontWeight="600">Evaluate fitness S</text>
        <text x="105" y="61" textAnchor="middle" fontSize="11" className="fill-slate-500">train XGBoost → val RMSE</text>

        <rect x="235" y="20" width="170" height="54" rx="10" className={box} />
        <text x="320" y="43" textAnchor="middle" fontWeight="600">Rank &amp; update best</text>
        <text x="320" y="61" textAnchor="middle" fontSize="11" className="fill-slate-500">bF, wF, DF, Xb</text>

        <rect x="450" y="20" width="170" height="54" rx="10" className="fill-green-50 stroke-green-600" />
        <text x="535" y="43" textAnchor="middle" fontWeight="600">Weights W</text>
        <text x="535" y="61" textAnchor="middle" fontSize="11" className="fill-slate-500">vein thickness 1 ± r·q</text>

        <rect x="20" y="200" width="170" height="64" rx="10" className={box} />
        <text x="105" y="224" textAnchor="middle" fontWeight="600">Relocate (p = z)</text>
        <text x="105" y="242" textAnchor="middle" fontSize="11" className="fill-slate-500">random position</text>
        <text x="105" y="256" textAnchor="middle" fontSize="11" className="fill-slate-500">exploration</text>

        <rect x="235" y="200" width="170" height="64" rx="10" className={box} />
        <text x="320" y="224" textAnchor="middle" fontWeight="600">Approach food</text>
        <text x="320" y="242" textAnchor="middle" fontSize="11" className="fill-slate-500">Xb + vb·(W·X_A − X_B)</text>
        <text x="320" y="256" textAnchor="middle" fontSize="11" className="fill-slate-500">prob. p = tanh|S − DF|</text>

        <rect x="450" y="200" width="170" height="64" rx="10" className={box} />
        <text x="535" y="224" textAnchor="middle" fontWeight="600">Wrap / contract</text>
        <text x="535" y="242" textAnchor="middle" fontSize="11" className="fill-slate-500">vc · X</text>
        <text x="535" y="256" textAnchor="middle" fontSize="11" className="fill-slate-500">otherwise</text>

        <text x="320" y="128" textAnchor="middle" fontWeight="600" className="fill-green-700">for each cell i</text>
        <text x="320" y="146" textAnchor="middle" fontSize="11" className="fill-slate-500">a = arctanh(1 − t/T), b = 1 − t/T shrink → exploitation</text>
      </g>
      <g className="stroke-slate-500" strokeWidth="1.5" fill="none" markerEnd="url(#arr)">
        <path d="M190 47 H233" />
        <path d="M405 47 H448" />
        <path d="M535 74 V110 H360" />
        <path d="M320 156 V198" />
        <path d="M280 156 L130 198" />
        <path d="M360 156 L510 198" />
        <path d="M20 232 H8 V47 H18" />
      </g>
      <text x="-6" y="140" fontSize="11" textAnchor="middle" className="fill-slate-500" transform="rotate(-90 -6 140)">next iteration t + 1</text>
    </svg>
  );
}


const TEAM = [
  { name: "Pramod M B", usn: "1BM23CD045" },
  { name: "Kashish Gaddigoudar", usn: "1BM24CD401" },
  { name: "Divan Singh", usn: "1BM23CD017" },
  { name: "Lakshaya Khandelwal", usn: "1BM23CD031" },
];

const PHASES:{ icon: IconName; title: string; formula: string; text: string }[] = [
  { icon: "network", title: "Vein thickness", formula: "W = 1 ± r·log10((bF − S)/(bF − wF) + 1)",
    text: "Cells are ranked by fitness. The better half gets thicker veins (W > 1), the worse half thinner ones (W < 1)." },
  { icon: "target", title: "Approach food", formula: "X = Xb + vb·(W·X_A − X_B)",
    text: "With probability p = tanh|S − DF| a cell moves around the best food found, guided by two random cells." },
  { icon: "shrink", title: "Wrap / contract", formula: "X = vc·X",
    text: "Otherwise the cell contracts, modelling the oscillating network. vb and vc shrink to 0 as iterations pass." },
  { icon: "shuffle", title: "Relocate", formula: "rand < z = 0.03",
    text: "Occasionally a cell jumps to a random position, which keeps exploring and avoids premature convergence." },
];

export default function Method() {
  return (
    <div className="space-y-8">
      <PageHeader eyebrow="Method" title="Nature-inspired tuning for EV charging demand">
        Given what is known when an electric vehicle plugs in — the station and port, plug type, time and day, and the driver's
        past behaviour — predict how much energy the session will draw. Good per-session forecasts help operators plan grid load,
        capacity and prices.
      </PageHeader>

      <section className="card border-l-4 border-l-green-600 px-5 py-4">
        <h2 className="panel-title">The biology</h2>
        <p className="mt-1.5 text-sm leading-relaxed text-zinc-700">
          <i>Physarum polycephalum</i> forages by spreading a network of tubes. Tubes leading to rich food carry more cytoplasm and
          thicken; tubes to poor food fade away. Placed on oat flakes arranged like the cities around Tokyo, a slime mould grew a
          network strikingly similar to the Tokyo rail system (Tero et al., <i>Science</i> 2010).
        </p>
      </section>

      <div>
        <h2 className="text-xl font-bold text-slate-900">The four moves of the Slime Mould Algorithm</h2>
        <p className="mt-1 text-sm text-slate-500">Li, Chen, Wang, Heidari &amp; Mirjalili, <i>Future Generation Computer Systems</i>, 2020 — implemented from scratch in NumPy.</p>
        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PHASES.map((p, i) => (
            <div key={p.title} className="card p-4">
              <div className="flex items-center justify-between">
                <span className="text-green-700"><Icon name={p.icon} /></span>
                <span className="num text-xs text-zinc-400">0{i + 1}</span>
              </div>
              <h3 className="mt-4 font-semibold text-slate-900">{p.title}</h3>
              <code className="mt-2 block rounded border border-zinc-200 bg-zinc-50 px-2 py-1 font-mono text-[11px] text-zinc-700">{p.formula}</code>
              <p className="mt-3 text-sm leading-relaxed text-slate-600">{p.text}</p>
            </div>
          ))}
        </div>
      </div>

      <Section title="One SMA iteration" sub="Every cell is evaluated, ranked and weighted, then takes one of three moves" icon="layers">
        <div className="flex justify-center"><SmaLoopDiagram /></div>
      </Section>

      <div className="grid gap-6 md:grid-cols-2">
        <Section title="Dataset" icon="database">
          <p className="text-sm leading-relaxed text-slate-600">
            City of Palo Alto, <i>Electric Vehicle Charging Station Usage (July 2011 – Dec 2020)</i> — one row per session.
            We keep 2012–2019 (2011 is sparse; 2020 is distorted by COVID-19) and split <b>chronologically</b> into 70% train,
            15% validation and 15% test, so the model is always evaluated on the future.
          </p>
        </Section>
        <Section title="The leakage rule" icon="shield">
          <p className="text-sm leading-relaxed text-slate-600">
            Columns known only after the session ends, or computed from energy — end date, duration, charging time, GHG and gasoline
            savings, fee, “ended by”, transaction date — are <b>dropped</b>. Driver and station history use only <i>earlier</i> sessions.
            Both rules are asserted in code and tested.
          </p>
        </Section>
      </div>

      <Section title="Team" icon="sparkles">
        <p className="text-sm text-slate-600">Bio-Inspired Algorithms (23DS7BSBIO) AAT · B.E. CSE (Data Science), BMSCE Bengaluru</p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {TEAM.map((m, i) => (
            <div key={m.usn} className="flex items-center gap-3 rounded border border-zinc-200 p-3">
              <span className="num flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-zinc-100 text-sm font-semibold text-zinc-600">{i + 1}</span>
              <div className="text-sm"><div className="font-medium text-zinc-900">{m.name}</div><div className="num text-xs text-zinc-500">{m.usn}</div></div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}
