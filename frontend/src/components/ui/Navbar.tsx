import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Workspace", end: true },
  { to: "/lab", label: "Model Lab", end: false },
];

/** Thin application bar. */
export default function Navbar() {
  return (
    <header className="flex h-11 shrink-0 items-center gap-6 border-b border-zinc-800 bg-zinc-900 px-4 text-sm text-zinc-300">
      <NavLink to="/" className="flex items-center gap-2 font-semibold text-white">
        <svg viewBox="0 0 16 16" className="h-4 w-4" aria-hidden="true">
          <path d="M9 1 3 9h4l-1 6 6-8H8l1-6z" fill="#22c55e" />
        </svg>
        EV Demand Planner
      </NavLink>
      <nav className="flex h-full">
        {LINKS.map((l) => (
          <NavLink key={l.to} to={l.to} end={l.end}
            className={({ isActive }) => `flex h-full items-center border-b-2 px-3 transition-colors ${
              isActive ? "border-green-500 text-white" : "border-transparent hover:text-white"}`}>
            {l.label}
          </NavLink>
        ))}
      </nav>
      <span className="ml-auto hidden text-xs text-zinc-500 sm:block">SMA-tuned XGBoost · BMSCE 23DS7BSBIO</span>
    </header>
  );
}
