import { NavLink, Outlet } from "react-router-dom";

const TABS = [
  { to: "/lab", label: "Performance" },
  { to: "/lab/optimization", label: "Optimisation replay" },
  { to: "/lab/method", label: "Method" },
];

/** Technical section: model comparison, optimiser internals and methodology. Scrolls inside the app frame. */
export default function LabLayout() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 px-4 sm:px-6">
          <span className="py-2.5 text-sm font-semibold text-zinc-900">Model Lab</span>
          <nav className="flex">
            {TABS.map((t) => (
              <NavLink key={t.to} to={t.to} end
                className={({ isActive }) => `border-b-2 px-3 py-2.5 text-sm transition-colors ${
                  isActive ? "border-zinc-900 font-medium text-zinc-900" : "border-transparent text-zinc-500 hover:text-zinc-900"}`}>
                {t.label}
              </NavLink>
            ))}
          </nav>
          <span className="ml-auto hidden py-2.5 text-xs text-zinc-500 md:block">How the estimates are made — models, SMA tuning and evaluation</span>
        </div>
      </div>
      <div className="mx-auto max-w-6xl space-y-6 px-4 py-6 sm:px-6">
        <Outlet />
      </div>
      <footer className="border-t border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap justify-between gap-2 px-4 py-3 text-2xs text-zinc-500 sm:px-6">
          <span>Bio-Inspired Algorithms (23DS7BSBIO) · BMSCE Bengaluru</span>
          <span>Data: City of Palo Alto EV charging sessions, 2012–2019</span>
        </div>
      </footer>
    </div>
  );
}
